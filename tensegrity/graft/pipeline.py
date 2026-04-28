"""
Hybrid Generation Pipeline: Tensegrity cognitive layer + LLM generation.

The pipeline runs Tensegrity and the LLM in tandem:
  1. Input text is parsed into a structured observation
  2. Tensegrity processes the observation (belief update, causal arena, memory)
  3. Tensegrity's belief state is projected into logit biases
  4. The LLM generates a response with those biases applied at every decode step
  5. If beliefs haven't converged, the LLM generates unbiased (graceful fallback)

This implements the "LLM as Broca's area" pattern:
  - Tensegrity does the reasoning (constraint satisfaction, hypothesis elimination)
  - The LLM does the talking (fluent language production under logit guidance)

Two modes:
  LOCAL:  Uses transformers LogitsProcessor for per-step dynamic biasing
  REMOTE: Uses static logit_bias on HF Inference API
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Any
import logging
import json

from tensegrity.broca.controller import CognitiveController
from tensegrity.broca.schemas import CognitiveAction, BeliefState
from tensegrity.graft.vocabulary import VocabularyGrounding
from tensegrity.graft.logit_bias import (
    TensegrityLogitsProcessor,
    StaticLogitBiasBuilder,
    GraftState,
)
from tensegrity.torch_device import inference_load_settings

logger = logging.getLogger(__name__)


class HybridPipeline:
    """
    Tensegrity+LLM hybrid generation.
    
    The cognitive layer resolves beliefs. The LLM narrates the resolution.
    Logit biases bridge the gap — no beliefs in the prompt, no reasoning
    delegated to the LLM.
    """
    
    def __init__(
        self,
        hypothesis_labels: List[str],
        hypothesis_keywords: Optional[Dict[str, List[str]]] = None,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        mode: str = "local",
        scale: float = 2.5,
        entropy_gate: float = 0.85,
        suppress_threshold: float = 0.01,
        async_graft: bool = True,
        # Semantic grounding is now the default (replaces brittle keyword-to-
        # token matching with frozen sbert phrase projection — the
        # "color wheel of meaning" instead of the "phrase book"). If sbert is
        # not available at runtime we fall back to keyword grounding.
        semantic_grounding: bool = True,
        semantic_embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        semantic_top_k: int = 32,
        semantic_threshold: Optional[float] = None,
    ):
        """
        Args:
            hypothesis_labels: List of hypothesis names
            hypothesis_keywords: {hyp: [keyword, ...]} for vocabulary grounding.
                               If None, auto-generated from labels.
            model_name: HF model ID
            mode: "local" (transformers + LogitsProcessor) or 
                  "remote" (HF Inference API + static logit_bias) or
                  "offline" (no LLM, template-based, for testing)
            scale: Logit bias magnitude
            entropy_gate: Convergence threshold for bias emission
            suppress_threshold: Below this probability → hard suppress
            async_graft: Local mode only — poll beliefs in a background thread for non-blocking decode
            semantic_grounding: If True, build grounding by frozen semantic
                phrase/token projection instead of exact keyword tokenization
            semantic_embedding_fn: Required when semantic_grounding=True; maps
                text to a fixed embedding vector without runtime training
            semantic_top_k: Semantic vocabulary tokens retained per hypothesis
            semantic_threshold: Optional minimum cosine similarity for semantic grounding
        """
        self.hypothesis_labels = hypothesis_labels
        self.model_name = model_name
        self.mode = mode
        self.scale = scale
        self.entropy_gate = entropy_gate
        self.suppress_threshold = suppress_threshold
        self.async_graft = async_graft
        self.semantic_grounding = semantic_grounding
        self.semantic_embedding_fn = semantic_embedding_fn
        self.semantic_top_k = semantic_top_k
        self.semantic_threshold = semantic_threshold
        
        # Initialize cognitive controller (template mode — no LLM for parsing)
        self.controller = CognitiveController(
            n_hypotheses=len(hypothesis_labels),
            hypothesis_labels=hypothesis_labels,
            use_llm=False,
        )
        
        # Vocabulary grounding and logit processor (initialized lazily)
        self._grounding: Optional[VocabularyGrounding] = None
        self._processor: Optional[TensegrityLogitsProcessor] = None
        self._static_builder: Optional[StaticLogitBiasBuilder] = None
        self._tokenizer = None
        self._model = None
        self._hypothesis_keywords = hypothesis_keywords
        
        # Generation tracking
        self._generations = 0
        self._graft_states: List[GraftState] = []

    def _label_phrases(self) -> Dict[str, List[str]]:
        phrases = {}
        for label in self.hypothesis_labels:
            parts = label.replace("_", " ").replace("-", " ").split()
            phrases[label] = [label.replace("_", " ").replace("-", " ")] + parts
        return phrases

    def _build_grounding(self) -> VocabularyGrounding:
        if self.semantic_grounding:
            embed = self.semantic_embedding_fn or self._default_sbert_embed_fn()
            if embed is not None:
                phrases = self._hypothesis_keywords or self._label_phrases()
                try:
                    return VocabularyGrounding.from_semantic_projection(
                        phrases,
                        self._tokenizer,
                        embedding_fn=embed,
                        top_k=self.semantic_top_k,
                        threshold=self.semantic_threshold,
                    )
                except Exception as e:
                    logger.warning(
                        "semantic grounding failed (%s); falling back to keyword grounding", e
                    )
        if self._hypothesis_keywords:
            return VocabularyGrounding.from_keywords(
                self._hypothesis_keywords, self._tokenizer)
        return VocabularyGrounding.from_labels_only(
            self.hypothesis_labels, self._tokenizer)

    def _default_sbert_embed_fn(self) -> Optional[Callable[[str], np.ndarray]]:
        """Build a frozen sbert embedding function. Used when the caller did
        not pass an explicit semantic_embedding_fn. No gradient flow.

        Uses a bulk-prefetch cache: on first invocation, batch-encodes the
        entire LLM vocabulary in one shot (a few seconds for ~128k tokens
        on CPU) so the per-token loop in SemanticProjectionLayer.from_tokenizer
        becomes a dict lookup instead of 128k individual sbert calls.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            logger.warning("sentence_transformers unavailable (%s); semantic grounding off", e)
            return None
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning("could not load sbert (%s); semantic grounding off", e)
            return None

        cache: Dict[str, np.ndarray] = {}
        bulk_done = [False]

        def _bulk_warm() -> None:
            if bulk_done[0] or self._tokenizer is None:
                return
            try:
                from tensegrity.graft.vocabulary import _token_texts_from_tokenizer
                texts = _token_texts_from_tokenizer(self._tokenizer)
            except Exception as e:
                logger.debug("vocab text extraction skipped: %s", e)
                bulk_done[0] = True
                return
            uniq = sorted({t for t in texts.values() if isinstance(t, str) and t})
            if not uniq:
                bulk_done[0] = True
                return
            logger.info("sbert bulk encode: %d unique vocab strings", len(uniq))
            vecs = model.encode(uniq, batch_size=256, show_progress_bar=False)
            for t, v in zip(uniq, vecs):
                cache[t] = np.asarray(v, dtype=np.float32)
            bulk_done[0] = True

        def embed(text: str) -> np.ndarray:
            if not bulk_done[0]:
                _bulk_warm()
            v = cache.get(text)
            if v is not None:
                return v
            v = np.asarray(model.encode([text], show_progress_bar=False)[0], dtype=np.float32)
            cache[text] = v
            return v

        return embed
    
    def _init_local(self):
        """Lazy initialization for local mode (loads model + tokenizer)."""
        if self._model is not None:
            return
        
        from transformers import AutoTokenizer, AutoModelForCausalLM

        dtype, device_map, move_to = inference_load_settings()
        logger.info(f"Loading {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if move_to is not None:
            self._model = self._model.to(move_to)
        
        # Build vocabulary grounding
        self._grounding = self._build_grounding()
        
        # Build logit processor
        self._processor = TensegrityLogitsProcessor(
            hypothesis_tokens=self._grounding.hypothesis_tokens,
            hypothesis_token_scores=self._grounding.hypothesis_token_scores,
            belief_fn=self._get_current_beliefs,
            vocab_size=self._tokenizer.vocab_size,
            scale=self.scale,
            suppress_threshold=self.suppress_threshold,
            entropy_gate=self.entropy_gate,
            async_beliefs=self.async_graft,
        )
        
        logger.info(f"Vocabulary grounding coverage: {self._grounding.coverage()}")
    
    def _init_remote(self):
        """Lazy initialization for remote mode."""
        if self._static_builder is not None:
            return
        
        from transformers import AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self._grounding = self._build_grounding()
        
        self._static_builder = StaticLogitBiasBuilder(
            hypothesis_tokens=self._grounding.hypothesis_tokens,
            hypothesis_token_scores=self._grounding.hypothesis_token_scores,
            scale=self.scale,
            suppress_threshold=self.suppress_threshold,
        )
    
    def _get_current_beliefs(self) -> Dict[str, float]:
        """
        Called by the LogitsProcessor at EVERY decode step.
        Returns current hypothesis posteriors from Tensegrity.
        """
        posteriors = {}
        for h in self.controller.belief_state.hypotheses:
            if h.id not in self.controller.belief_state.eliminated_hypotheses:
                posteriors[h.description] = h.probability
        return posteriors
    
    def process_observation(self, text: str) -> Dict[str, Any]:
        """
        Feed an observation to the cognitive layer.
        Updates beliefs, memory, causal arena — everything.
        
        Returns the cognitive state after processing.
        """
        return self.controller.step(text)
    
    def generate_response(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """
        Generate a response with Tensegrity logit biases.
        
        The cognitive layer has already processed observations and formed beliefs.
        Now the LLM generates language guided by those beliefs.
        
        Args:
            prompt: The generation prompt (should describe what to output)
            max_tokens: Maximum tokens to generate
        
        Returns:
            {
                "text": generated text,
                "graft_state": diagnostics about the logit bias injection,
                "beliefs": current hypothesis posteriors,
                "mode": "grafted" or "fallback"
            }
        """
        if self.mode == "offline":
            return self._generate_offline(prompt)
        elif self.mode == "local":
            return self._generate_local(prompt, max_tokens)
        elif self.mode == "remote":
            return self._generate_remote(prompt, max_tokens)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _generate_local(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Generate with per-step logit bias injection via LogitsProcessor."""
        self._init_local()
        
        from transformers import LogitsProcessorList
        
        # Build the prompt. Newer transformers may return a BatchEncoding
        # rather than a bare tensor — unwrap to the input_ids tensor before
        # passing to model.generate.
        import torch as _torch
        messages = [{"role": "user", "content": prompt}]
        encoded = self._tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if isinstance(encoded, _torch.Tensor):
            input_ids = encoded
        elif hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
        elif isinstance(encoded, dict) and "input_ids" in encoded:
            input_ids = encoded["input_ids"]
        else:
            input_ids = _torch.as_tensor(encoded)

        if hasattr(self._model, 'device'):
            input_ids = input_ids.to(self._model.device)

        # Generate with Tensegrity logit processor
        self._processor._step_count = 0  # Reset step counter

        outputs = self._model.generate(
            input_ids,
            logits_processor=LogitsProcessorList([self._processor]),
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        self._generations += 1
        graft_state = {
            "step": self._processor.state.step,
            "bias_emitted": self._processor.state.bias_emitted,
            "belief_entropy": self._processor.state.belief_entropy,
            "convergence_met": self._processor.state.convergence_met,
            "max_bias_magnitude": self._processor.state.max_bias_magnitude,
        }
        
        return {
            "text": text,
            "graft_state": graft_state,
            "beliefs": self._get_current_beliefs(),
            "mode": "grafted" if self._processor.state.bias_emitted else "fallback",
        }
    
    def _generate_remote(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Generate with static logit bias via HF Inference API."""
        self._init_remote()
        
        import os
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN"),
        )
        
        # Build static bias from current beliefs
        posteriors = self._get_current_beliefs()
        logit_bias = self._static_builder.build(posteriors)
        
        # Check convergence gate
        probs = np.array(list(posteriors.values()))
        probs = probs[probs > 0]
        if len(probs) > 1:
            entropy = -np.sum(probs * np.log(probs)) / np.log(len(probs))
        else:
            entropy = 0.0
        
        converged = entropy < self.entropy_gate
        
        # Only apply bias if converged
        kwargs = {}
        if converged and logit_bias:
            kwargs["logit_bias"] = logit_bias
        
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            **kwargs,
        )
        
        text = completion.choices[0].message.content
        
        return {
            "text": text,
            "graft_state": {"converged": converged, "entropy": entropy, "n_biased_tokens": len(logit_bias)},
            "beliefs": posteriors,
            "mode": "grafted" if converged else "fallback",
        }
    
    def _generate_offline(self, prompt: str) -> Dict[str, Any]:
        """
        Offline generation: no LLM, use cognitive state directly.
        
        Demonstrates that the cognitive layer can resolve the answer
        without any language model at all.
        """
        posteriors = self._get_current_beliefs()
        
        if posteriors:
            winner = max(posteriors, key=posteriors.get)
            confidence = posteriors[winner]
            
            # Compute convergence
            probs = np.array(list(posteriors.values()))
            probs = probs[probs > 0]
            if len(probs) > 1:
                entropy = -np.sum(probs * np.log(probs)) / np.log(len(probs))
            else:
                entropy = 0.0
            
            if confidence > 0.4:
                text = f"The answer is {winner}."
            elif confidence > 0.25:
                runner_up = sorted(posteriors, key=posteriors.get, reverse=True)[1] if len(posteriors) > 1 else "unknown"
                text = f"Most likely {winner}, though {runner_up} is also possible."
            else:
                text = "I don't have enough evidence to determine the answer yet."
        else:
            text = "No hypotheses to evaluate."
            entropy = 1.0
            winner = None
            confidence = 0.0
        
        return {
            "text": text,
            "graft_state": {"converged": entropy < self.entropy_gate, "entropy": float(entropy)},
            "beliefs": posteriors,
            "mode": "offline",
        }
    
    def run_scenario(self, clues: List[str], 
                     generation_prompt: str = "Based on all the evidence, what is the answer?",
                     verbose: bool = True) -> Dict[str, Any]:
        """
        Run a full scenario: process clues then generate a response.
        
        This is the end-to-end pipeline:
          1. Feed clues one by one → cognitive layer processes each
          2. After all clues, generate a response with logit biases
        
        Returns full results including belief trajectory and generation output.
        """
        belief_trajectory = []
        
        for i, clue in enumerate(clues):
            result = self.process_observation(clue)
            
            posteriors = self._get_current_beliefs()
            belief_trajectory.append({
                "turn": i + 1,
                "clue": clue,
                "posteriors": dict(posteriors),
                "tension": result["perception"]["tension"],
                "free_energy": result["perception"]["free_energy"],
            })
            
            if verbose:
                top = max(posteriors, key=posteriors.get) if posteriors else "?"
                top_p = posteriors.get(top, 0)
                print(f"  Clue {i+1}: \"{clue[:60]}...\"" if len(clue) > 60 else f"  Clue {i+1}: \"{clue}\"")
                print(f"    → Top: {top} (p={top_p:.3f}), entropy={belief_trajectory[-1].get('tension', '?'):.3f}")
        
        # Generate response
        if verbose:
            print(f"\n  Generating response...")
        
        gen_result = self.generate_response(generation_prompt)
        
        if verbose:
            print(f"  Mode: {gen_result['mode']}")
            print(f"  Output: \"{gen_result['text']}\"")
            print(f"  Final beliefs: {json.dumps({k: round(v, 3) for k, v in sorted(gen_result['beliefs'].items(), key=lambda x: x[1], reverse=True)[:4]})}")
        
        return {
            "belief_trajectory": belief_trajectory,
            "generation": gen_result,
            "final_beliefs": gen_result["beliefs"],
        }
    
    @property
    def state_summary(self) -> str:
        return self.controller.get_state_summary()


