---
license: apache-2.0
tags:
- active-inference
- causal-inference
- bayesian
- free-energy-principle
- non-gradient
- cognitive-architecture
- morton-code
- belief-propagation
- hopfield-network
---

# Tensegrity: A Non-Gradient Cognitive Architecture

**No backpropagation. No gradient descent. No optimizer state.**

Tensegrity is a cognitive architecture where structural integrity comes from the **tension between competing causal models**, not from gradient-based optimization. It implements the actual mathematics of Friston, Pearl, Markov, Bayes, and Zipf — not as metaphors, but as computational machinery.

## The Core Idea

Instead of minimizing a loss function via SGD, Tensegrity minimizes **variational free energy** via **fixed-point iteration** (belief propagation on factor graphs). Multiple structural causal models compete to explain observations, and the tension between them drives learning and exploration.

```
F = D_KL[q(s) || p(s)] - E_q[ln p(o | s)]
    ─────────────────    ─────────────────
       Complexity            Accuracy
```

This is minimized by coordinate ascent — each step is a matrix multiply + softmax normalization. The entire system converges without computing a single gradient.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MARKOV BLANKET                        │
│  ┌──────────┐                          ┌──────────┐    │
│  │ SENSORY  │  Morton-coded input       │  ACTIVE  │    │
│  │  STATES  │ ─────────────────────┐    │  STATES  │    │
│  └──────────┘                      │    └────┬─────┘    │
│       │                            │         │          │
│       ▼                            │         │          │
│  ┌──────────────────────────┐      │         │          │
│  │     BELIEF STATES        │      │         │          │
│  │  Q(s) over hidden states │◄─────┘         │          │
│  │  Updated via VFE min     │                │          │
│  └─────────┬────────────────┘                │          │
│            │                                 │          │
│            ▼                                 │          │
│  ┌──────────────────────────┐                │          │
│  │   CAUSAL ARENA           │                │          │
│  │  M₁ vs M₂ vs ... vs Mₖ  │────────────────┘          │
│  │  SCMs compete via F      │                           │
│  └─────────┬────────────────┘                           │
│            │                                            │
│            ▼                                            │
│  ┌──────────────────────────────────────┐               │
│  │         MEMORY SYSTEMS               │               │
│  │  Epistemic │ Episodic │ Associative  │               │
│  │  (beliefs) │ (traces) │ (Hopfield)   │               │
│  │  Zipf-weighted access priority       │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Morton-Coded Sensory Input
Any modality (image, audio, text, sensor data) is encoded via **Z-order curves** (Morton codes) into a single integer that preserves spatial locality. The system literally doesn't know what modality it's processing — it's all just Morton codes.

### 2. Free Energy Engine (Friston)
Implements the discrete POMDP active inference loop:
- **Perception**: Fixed-point iteration on `q(s)` until convergence (~16 iterations, no gradients)
- **Planning**: Expected Free Energy `G_π = ambiguity + risk` for each policy
- **Action**: Softmax over `-γ·G_π` (Boltzmann policy selection)

### 3. Causal Arena (Pearl)
Multiple Structural Causal Models compete to explain observations:
- **Rung 1 — Association**: `P(Y | X=x)` via standard conditioning
- **Rung 2 — Intervention**: `P(Y | do(X=x))` via graph surgery (mutilated DAG)
- **Rung 3 — Counterfactual**: `P(Y_x | X=x', Y=y')` via abduction → intervention → prediction

The **tension** is the entropy of the posterior over models: high entropy = competing explanations, low entropy = one model dominates.

### 4. Belief Propagation (Markov/Bayes)
Sum-product algorithm on factor graphs with loopy BP + damping for cyclic structures. This is the shared computational primitive for:
- State inference (perception)
- Policy evaluation (planning)
- Model comparison (arena)

### 5. Memory Systems (Zipf)

| Memory | What it stores | Update rule | Retrieval |
|--------|---------------|-------------|-----------|
| **Epistemic** | Dirichlet-parameterized beliefs (A, B, C, D matrices) | Bayesian counting | Zipf-weighted by access frequency |
| **Episodic** | Temporal experience traces with context vectors | Context drift (TCM) | Cosine similarity + recency + Zipf |
| **Associative** | Modern Hopfield network (exponential capacity) | Pattern append | Energy minimization (no gradients) |

All three follow **Zipf's law**: frequently accessed memories are cheapest to retrieve, creating self-reinforcing power-law access patterns.

## Quick Start

```python
from tensegrity import TensegrityAgent
import numpy as np

# Create agent
agent = TensegrityAgent(
    n_states=16,
    n_observations=32,
    n_actions=4,
    sensory_dims=4,     # 4D input (any modality)
    sensory_bits=8,     # 256 levels per dimension
    precision=4.0,      # Inverse temperature
    zipf_exponent=1.0,  # Power-law steepness
)

# Perception-action loop
for t in range(100):
    raw_observation = np.random.randn(4)  # Any modality, any shape
    
    # Perceive: Morton encode → Free energy minimize → Update beliefs
    result = agent.perceive(raw_observation)
    
    # Act: Expected free energy → Policy selection
    action = agent.act()
    
    print(f"t={t}: F={result['free_energy']:.2f}, "
          f"tension={result['arena']['tension']:.3f}, "
          f"surprise={result['surprise']:.2f}")

# Introspect
state = agent.introspect()
print(f"Arena winner: {state['arena']['current_winner']}")
print(f"Epistemic entropy: {state['epistemic_memory']['entropy']}")

# Counterfactual reasoning
cf = agent.counterfactual(
    evidence={'state': 0, 'observation': 5},
    intervention={'state': 3},
    query=['observation']
)
```

## What "Tension" Means Here

The tension is not a metaphor. It's a measurable quantity:

```
tension = H[P(M₁, M₂, ..., Mₖ | data)] / log(K)
```

Where `H` is Shannon entropy and `K` is the number of competing causal models. 

- **Tension = 1.0**: All models equally likely. Maximum uncertainty about causal structure.
- **Tension = 0.0**: One model dominates. The system has "decided" which causal story is correct.
- **Tension > 0.5**: The system should explore (epistemic actions) to resolve the competition.

The tension drives the system to **choose experiments** that would maximally discriminate between competing models. This is the epistemic component of Expected Free Energy — it's information-seeking behavior emerging from pure logic.

## What's NOT Here

- ❌ No neural networks
- ❌ No backpropagation
- ❌ No gradient descent
- ❌ No loss functions
- ❌ No optimizer state (Adam, SGD, etc.)
- ❌ No learned weights
- ❌ No training epochs

## What IS Here

- ✅ Variational free energy minimization (coordinate ascent)
- ✅ Bayesian belief updating (Dirichlet counting)
- ✅ Belief propagation (sum-product algorithm)
- ✅ Structural causal models (Pearl's do-calculus)
- ✅ Counterfactual reasoning (abduction → intervention → prediction)
- ✅ Modern Hopfield networks (energy-based associative memory)
- ✅ Temporal Context Model (episodic memory)
- ✅ Morton codes (space-filling curve encoding)
- ✅ Zipf-distributed memory access (power-law priority)

## Theoretical Foundations

| Component | Theory | Key Paper |
|-----------|--------|-----------|
| Free Energy Engine | Active Inference / FEP | Friston et al., "Active Inference" (arXiv:2006.04120) |
| Causal Arena | Structural Causal Models | Pearl, "Causality" (2009) |
| Belief Propagation | Sum-Product Algorithm | Kschischang et al. (2001) |
| Associative Memory | Modern Hopfield Networks | Ramsauer et al. (arXiv:2008.02217) |
| Episodic Memory | Temporal Context Model | Howard & Kahana (2002) |
| Morton Encoding | Z-Order Curves | Morton (1966) |
| Zipf Priority | Power-Law Access | Anderson & Schooler (1991) |

## Dependencies

- `numpy` — Array operations
- `scipy` — Softmax, digamma, special functions
- `networkx` — Graph operations for causal DAGs

## License

Apache 2.0
