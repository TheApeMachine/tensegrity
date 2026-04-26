"""
Vocabulary Grounding: Maps hypotheses to token sets in the LLM's vocabulary.

Each hypothesis in the cognitive layer gets associated with a set of tokens
in the LLM's vocabulary. When a hypothesis has high posterior probability,
those tokens get boosted. When a hypothesis is eliminated, those tokens
get suppressed.

The grounding is built once per hypothesis set from keyword lists.
Each keyword is tokenized with both bare and space-prefixed variants
to handle sentence-initial and mid-sentence positions.

This is the bridge between Tensegrity's discrete Bayesian states
and the LLM's continuous logit space.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


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
    
    # Inverse map: {token_id: list of hypothesis_ids it belongs to}
    token_to_hypotheses: Dict[int, List[str]] = field(default_factory=dict)
    
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
    
    def get_token_ids(self, hypothesis_id: str) -> Set[int]:
        """Get all token IDs associated with a hypothesis."""
        return self.hypothesis_tokens.get(hypothesis_id, set())
    
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
