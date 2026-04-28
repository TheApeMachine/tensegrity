---
license: apache-2.0
tags:
- active-inference
- causal-inference
- bayesian
- free-energy-principle
- non-gradient
- cognitive-architecture
- predictive-coding
- fhrr
- hopfield-network
---

# Tensegrity: Non-Gradient Cognitive Architecture

Tensegrity is centered on a unified energy landscape:

```
FHRR encoding -> hierarchical predictive coding -> Hopfield memory
              -> optional causal energy terms -> Broca/LLM graft
```

The language model is treated as a linguistic interface. The cognitive layer
owns belief revision, causal competition, memory, and action selection; the LLM
only verbalizes under optional logit guidance.

## Current API

Use the V2 unified field by default:

```python
from tensegrity import UnifiedField

field = UnifiedField(
    obs_dim=128,
    hidden_dims=[64, 16],
    fhrr_dim=1024,
    ngc_settle_steps=15,
)

cycle = field.observe(
    {"object": "ball", "color": "red", "location": "table"},
    input_type="bindings",
)

print(cycle["energy"].total)
print(field.predict())
```

The old Morton/POMDP frontend is still available for migration and baselines:

```python
from tensegrity.legacy.v1 import TensegrityAgent, MortonEncoder, MarkovBlanket
```

Compatibility shims remain under `tensegrity.core.agent`,
`tensegrity.core.morton`, and `tensegrity.core.blanket`, but those modules emit
deprecation warnings and are not part of the primary export surface.

## Core Pieces

| Package                                 | Role                                                           |
|-----------------------------------------|----------------------------------------------------------------|
| `tensegrity.core` / `tensegrity.engine` | V2 unified field, FHRR, NGC, Hopfield memory, causal energy    |
| `tensegrity.graft`                      | Broca-style LLM graft with keyword or semantic token grounding |
| `tensegrity.causal`                     | Pearl SCMs, do-calculus, counterfactuals                       |
| `tensegrity.memory`                     | Epistemic, episodic, and associative memory baselines          |
| `tensegrity.legacy.v1`                  | Morton-coded Markov blanket and flat POMDP agent               |

## Unified Energy

The unified field decomposes total energy into local prediction-error terms:

```
E_total = E_perception + E_memory + E_causal
```

Where:

```text
E_perception = hierarchical predictive-coding residuals
E_memory     = Modern Hopfield retrieval energy
E_causal     = SCM prediction error over causal variables
```

All updates are local fixed-point or Hebbian-style operations. There is no
runtime backpropagation loop or optimizer state in the cognitive architecture.

## Causal Topology Mapping

`tensegrity.engine.causal_energy.TopologyMapper` makes the Pearl/Friston bridge
explicit. It projects an arbitrary acyclic SCM graph into NGC-compatible layers:

- direct layer-to-layer causal edges become top-down predictions;
- bypass edges receive relay nodes for skipped hierarchy levels;
- same-layer or inverted edges receive virtual parent nodes one layer above the
  endpoints, turning lateral causal structure into shared vertical dependency.

## Semantic Grafting

`VocabularyGrounding.from_keywords(...)` remains as a deterministic baseline.
For less brittle grounding, `VocabularyGrounding.from_semantic_projection(...)`
uses frozen phrase/token embeddings and cosine proximity to build weighted
token sets without runtime gradient training.

## What Is Not Here

- No SGD or Adam-style optimizer state in the cognitive loop
- No backpropagation-through-time training loop
- No prompt-only delegation of reasoning to the LLM

## What Is Here

- FHRR compositional observation encoding
- Hierarchical predictive coding
- Modern Hopfield memory
- Structural causal models and counterfactuals
- Explicit SCM-to-NGC topology mapping
- Keyword and semantic LLM logit grounding
- Legacy V1 baselines for comparison

## Dependencies

- `numpy`
- `scipy`
- `networkx`
- `pydantic`
- `torch`
- `transformers`
- `sentence-transformers`

## License

Apache 2.0
