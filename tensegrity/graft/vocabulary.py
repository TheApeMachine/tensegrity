"""
Vocabulary Grounding: Maps hypotheses to token sets in the LLM's vocabulary.

Each hypothesis in the cognitive layer gets associated with a set of tokens
in the LLM's vocabulary. When a hypothesis has high posterior probability,
those tokens get boosted. When a hypothesis is eliminated, those tokens
get suppressed.

The baseline grounding is built once per hypothesis set from keyword lists.
Each keyword is tokenized with both bare and space-prefixed variants to handle
sentence-initial and mid-sentence positions.

For less brittle grounding, ``from_semantic_projection`` builds a non-gradient
projection layer over frozen phrase/token vectors. Hypotheses become continuous
concept vectors, vocabulary items become continuous token vectors, and token
selection is based on cosine proximity rather than exact lexical matches.

This is the bridge between Tensegrity's discrete Bayesian states
and the LLM's continuous logit space.
"""

import re
from typing import Callable, Dict, Iterable, List, Set, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


EmbeddingFn = Callable[[str], np.ndarray]


def _clean_token_text(token: str) -> str:
    """Normalize common tokenizer word-boundary markers into plain text."""
    text = str(token)
    text = text.replace("Ġ", " ").replace("▁", " ").replace("</w>", "")
    text = re.sub(r"^##", "", text)
    return text.strip()


def _label_phrases(label: str) -> List[str]:
    parts = label.replace("_", " ").replace("-", " ").split()
    phrases = [label.replace("_", " ").replace("-", " ")]
    phrases.extend(parts)
    return [p for p in phrases if p]


def _as_unit_vector(value: np.ndarray) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def _mean_unit_vector(vectors: Iterable[np.ndarray]) -> np.ndarray:
    unit_vectors = [_as_unit_vector(v) for v in vectors]
    unit_vectors = [v for v in unit_vectors if v.size and np.linalg.norm(v) > 1e-12]
    if not unit_vectors:
        return np.array([], dtype=np.float64)
    return _as_unit_vector(np.mean(np.stack(unit_vectors, axis=0), axis=0))


def _token_texts_from_tokenizer(tokenizer) -> Dict[int, str]:
    """Return cleaned token text keyed by token id for common tokenizer APIs."""
    token_texts: Dict[int, str] = {}
    if hasattr(tokenizer, "get_vocab"):
        for raw_text, tid in tokenizer.get_vocab().items():
            clean = _clean_token_text(raw_text)
            if clean:
                token_texts[int(tid)] = clean
        return token_texts

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return token_texts
    for tid in range(vocab_size):
        try:
            clean = _clean_token_text(decode([tid]))
        except Exception:
            continue
        if clean:
            token_texts[tid] = clean
    return token_texts


@dataclass
class SemanticProjectionLayer:
    """
    Frozen semantic bridge from cognitive vectors or phrases to vocabulary IDs.

    ``embedding_fn`` is intentionally supplied from the outside. It can wrap a
    sentence-transformer, a model's input embedding table, an offline linear
    probe, or FHRR phrase vectors. During inference this class performs only
    normalization, matrix multiplication, and top-k selection.
    """

    token_vectors: Dict[int, np.ndarray]
    token_texts: Dict[int, str] = field(default_factory=dict)
    projection_matrix: Optional[np.ndarray] = None

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        embedding_fn: EmbeddingFn,
        projection_matrix: Optional[np.ndarray] = None,
        token_texts: Optional[Dict[int, str]] = None,
    ) -> "SemanticProjectionLayer":
        texts = token_texts or _token_texts_from_tokenizer(tokenizer)
        token_vectors: Dict[int, np.ndarray] = {}
        for tid, text in texts.items():
            try:
                vec = _as_unit_vector(embedding_fn(text))
            except Exception:
                continue
            if vec.size and np.linalg.norm(vec) > 1e-12:
                token_vectors[int(tid)] = vec
        return cls(
            token_vectors=token_vectors,
            token_texts={int(k): v for k, v in texts.items()},
            projection_matrix=None if projection_matrix is None else np.asarray(projection_matrix),
        )

    def project_state(
        self,
        state_vector: np.ndarray,
        top_k: int = 32,
        threshold: Optional[float] = None,
    ) -> Dict[int, float]:
        """
        Project a cognitive state vector into vocabulary-token similarity scores.

        If ``projection_matrix`` is set, it is applied once before scoring. This
        covers the frozen linear-probe case without runtime gradients.
        """
        vec = np.asarray(state_vector, dtype=np.float64).ravel()
        if self.projection_matrix is not None:
            vec = self.projection_matrix @ vec
        return self._top_token_scores(_as_unit_vector(vec), top_k=top_k, threshold=threshold)

    def project_phrase_vector(
        self,
        phrase_vector: np.ndarray,
        top_k: int = 32,
        threshold: Optional[float] = None,
    ) -> Dict[int, float]:
        """Score vocabulary tokens against an already encoded phrase/concept vector."""
        return self._top_token_scores(_as_unit_vector(phrase_vector), top_k=top_k, threshold=threshold)

    def _top_token_scores(
        self,
        concept: np.ndarray,
        top_k: int,
        threshold: Optional[float],
    ) -> Dict[int, float]:
        if not concept.size or not self.token_vectors:
            return {}
        scores: List[Tuple[int, float]] = []
        for tid, vec in self.token_vectors.items():
            if vec.shape != concept.shape:
                continue
            score = float(np.dot(concept, vec))
            if threshold is None or score >= threshold:
                scores.append((tid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k > 0:
            scores = scores[:top_k]
        return {tid: score for tid, score in scores}


@dataclass
class VocabularyGrounding:
    """
    Maps hypothesis labels to sets of token IDs in the LLM vocabulary.
    
    Built once, used every decode step by the LogitsProcessor.
    """
    # {hypothesis_id: set of token IDs}
    hypothesis_tokens: Dict[str, Set[int]] = field(default_factory=dict)
    
    # {hypothesis_id: list of grounding keywords}
    hypothesis_keywords: Dict[str, List[str]] = field(default_factory=dict)

    # {hypothesis_id: {token_id: semantic proximity in [roughly -1, 1]}}
    hypothesis_token_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    
    # Inverse map: {token_id: list of hypothesis_ids it belongs to}
    token_to_hypotheses: Dict[int, List[str]] = field(default_factory=dict)

    # Optional frozen projection layer used to build semantic token scores.
    semantic_projection: Optional[SemanticProjectionLayer] = None
    
    vocab_size: int = 0
    
    @classmethod
    def from_keywords(cls, hypothesis_keywords: Dict[str, List[str]], 
                      tokenizer) -> 'VocabularyGrounding':
        """
        Build grounding from keyword lists.
        
        Args:
            hypothesis_keywords: {hyp_id: ["word1", "word2", ...]}
            tokenizer: HuggingFace tokenizer (any tokenizer with .encode())
        
        Example:
            {
                "parrot": ["parrot", "bird", "feather", "beak", "wings", "fly"],
                "snake": ["snake", "reptile", "scales", "slither", "venom"],
            }
        """
        grounding = cls()
        grounding.vocab_size = tokenizer.vocab_size
        grounding.hypothesis_keywords = hypothesis_keywords
        
        for hyp_id, keywords in hypothesis_keywords.items():
            token_set = set()
            for word in keywords:
                # Tokenize with and without space prefix
                for variant in [word, f" {word}", word.capitalize(), f" {word.capitalize()}"]:
                    try:
                        ids = tokenizer.encode(variant, add_special_tokens=False)
                        token_set.update(ids)
                    except Exception:
                        continue
            
            grounding.hypothesis_tokens[hyp_id] = token_set
            
            # Build inverse map
            for tid in token_set:
                if tid not in grounding.token_to_hypotheses:
                    grounding.token_to_hypotheses[tid] = []
                grounding.token_to_hypotheses[tid].append(hyp_id)
        
        return grounding
    
    @classmethod
    def from_labels_only(cls, hypothesis_labels: List[str], 
                         tokenizer) -> 'VocabularyGrounding':
        """
        Minimal grounding: just tokenize the hypothesis labels themselves.
        Quick but narrow — use from_keywords for better coverage.
        """
        keywords = {}
        for label in hypothesis_labels:
            # Split compound labels (e.g., "memory_leak" → ["memory", "leak"])
            parts = label.replace("_", " ").replace("-", " ").split()
            keywords[label] = [label] + parts
        
        return cls.from_keywords(keywords, tokenizer)

    @classmethod
    def from_semantic_projection(
        cls,
        hypothesis_phrases: Dict[str, List[str]],
        tokenizer,
        embedding_fn: EmbeddingFn,
        *,
        projection_matrix: Optional[np.ndarray] = None,
        token_texts: Optional[Dict[int, str]] = None,
        top_k: int = 32,
        threshold: Optional[float] = None,
    ) -> 'VocabularyGrounding':
        """
        Build grounding by semantic proximity instead of exact keyword matches.

        Args:
            hypothesis_phrases: {hyp_id: ["phrase", ...]} concept descriptions.
            tokenizer: tokenizer with ``vocab_size`` and preferably ``get_vocab``.
            embedding_fn: frozen text embedding function. No training is run here.
            projection_matrix: optional frozen linear map from cognitive-state
                vectors into the embedding space used by ``embedding_fn``.
            token_texts: optional explicit {token_id: token_text} inventory.
            top_k: maximum vocabulary tokens retained per hypothesis.
            threshold: minimum cosine similarity. ``None`` keeps the best top_k.
        """
        projection = SemanticProjectionLayer.from_tokenizer(
            tokenizer,
            embedding_fn=embedding_fn,
            projection_matrix=projection_matrix,
            token_texts=token_texts,
        )
        grounding = cls()
        grounding.vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        grounding.hypothesis_keywords = {
            hyp_id: list(phrases) for hyp_id, phrases in hypothesis_phrases.items()
        }
        grounding.semantic_projection = projection

        for hyp_id, phrases in hypothesis_phrases.items():
            concept_phrases = _label_phrases(hyp_id)
            concept_phrases.extend(str(p) for p in phrases)
            concept_vector = _mean_unit_vector(
                embedding_fn(phrase) for phrase in concept_phrases if str(phrase).strip()
            )
            token_scores = projection.project_phrase_vector(
                concept_vector,
                top_k=top_k,
                threshold=threshold,
            )
            grounding.hypothesis_token_scores[hyp_id] = token_scores
            grounding.hypothesis_tokens[hyp_id] = set(token_scores)

            for tid in token_scores:
                grounding.token_to_hypotheses.setdefault(tid, []).append(hyp_id)

        return grounding
    
    def get_token_ids(self, hypothesis_id: str) -> Set[int]:
        """Get all token IDs associated with a hypothesis."""
        return self.hypothesis_tokens.get(hypothesis_id, set())

    def get_token_scores(self, hypothesis_id: str) -> Dict[int, float]:
        """Get weighted token scores associated with a hypothesis."""
        scores = self.hypothesis_token_scores.get(hypothesis_id)
        if scores is not None:
            return scores
        return {tid: 1.0 for tid in self.get_token_ids(hypothesis_id)}
    
    def coverage(self) -> Dict[str, int]:
        """How many tokens are grounded per hypothesis."""
        return {h: len(toks) for h, toks in self.hypothesis_tokens.items()}
    
    def overlap(self) -> Dict[str, List[str]]:
        """Find tokens shared between hypotheses (potential confusion points)."""
        overlaps = {}
        hyps = list(self.hypothesis_tokens.keys())
        for i, h1 in enumerate(hyps):
            for h2 in hyps[i+1:]:
                shared = self.hypothesis_tokens[h1] & self.hypothesis_tokens[h2]
                if shared:
                    overlaps[f"{h1}↔{h2}"] = list(shared)
        return overlaps


