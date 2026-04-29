The ideal architecture is not “LLM plus Tensegrity.” It is one agent with the LLM as a language organ inside a persistent predictive system.

**Core Principle**

Tensegrity should own the world model, memory, beliefs, causal structure, uncertainty, and action selection.

The LLM should do three jobs only:

1. Parse language into structured observations.
2. Provide linguistic likelihoods/logits as sensory evidence.
3. Verbalize or emit the chosen action under live Tensegrity constraint.

It should not be an external baseline scorer that gets combined afterward.

**Ideal Loop**

1. A benchmark item arrives as environmental input.

2. Broca parses the prompt into typed entities, relations, roles, quantities, negations, and question type.

3. FHRR encodes that structure into compositional vectors.

4. Persistent episodic memory retrieves similar prior problems.

5. Persistent associative memory retrieves similar latent states.

6. Epistemic memory supplies learned priors over observations, states, actions, and task patterns.

7. Causal memory retrieves or proposes relevant SCMs.

8. Predictive coding receives the current structured observation plus retrieved memories.

9. NGC settles the latent state by minimizing prediction error.

10. Each candidate answer is treated as a hypothesis/action, not as a separate external score.

11. The LLM exposes answer-token likelihoods as linguistic sensory evidence.

12. Those likelihoods enter the belief update as one evidence channel.

13. NGC tests each candidate by asking: “If this were true, how well would it predict the prompt?”

14. The causal arena asks: “Which candidate best fits the causal structure of the situation?”

15. Memory asks: “Which candidate resembles past successful explanations in similar contexts?”

16. Active inference integrates all evidence into one posterior over candidate actions.

17. The agent selects the answer/action that minimizes expected free energy.

18. The LLM emits the answer under live logit grafting from the selected belief state.

19. The benchmark reveals success or failure.

20. That outcome becomes feedback, not just a metric.

21. The agent stores the episode persistently.

22. Epistemic counts update.

23. Predictive-coding weights adapt.

24. Causal model weights update.

25. Memory consolidates the experience.

26. Future benchmark items start from the improved persistent state.

**Where Each Part Fits**

Broca / LLM:
The LLM is the interface between language and cognition. It parses raw text into structure, provides token/logit evidence, and verbalizes conclusions. It should be inside the loop, not outside it.

FHRR:
FHRR is the symbolic-continuous binding layer. It represents “object has property,” “choice explains prompt,” “cause leads to effect,” and similar structured relations as vectors the field can process.

Predictive Coding / NGC:
This should be the central engine. Every hypothesis makes predictions. NGC measures prediction error. Beliefs change because some hypotheses explain the observation better than others.

Hopfield / Associative Memory:
This stores attractor states: recurring problem structures, answer patterns, latent explanations, and successful interpretations.

Episodic Memory:
This stores whole benchmark experiences: prompt structure, retrieved context, chosen answer, posterior, prediction errors, outcome, and surprise.

Epistemic Memory:
This is the learned statistical substrate: Dirichlet counts, likelihoods, transition expectations, and action-outcome priors.

Causal SCM Arena:
This stores competing causal explanations. It should not be rebuilt fresh for every item. It should persist, specialize, gain/lose credibility, and support counterfactual tests.

Active Inference:
This is the decision policy. It decides whether to answer, defer, ask, explore, retrieve more memory, or propose/update a causal model.

Logit Graft:
The graft should be live at generation time. It should reflect the current posterior, not a post-hoc additive score.

**What Is Not Fully Realized Yet**

1. The benchmark still performs late fusion: `LLM score + λ*Tensegrity score`.

2. Broca parsing is disabled in the benchmark path.

3. LLM logits are not treated as sensory evidence inside the cognitive loop.

4. The graft is not the primary benchmark decision path.

5. Memory is not persisted across runs.

6. Benchmark feedback is not used as learning signal.

7. Per-choice SCMs are rebuilt per item instead of becoming durable causal knowledge.

8. Predictive coding is used as a scorer, not as the global organizing loop.

9. There are multiple partially overlapping scoring paths.

10. There is no single serialized agent state.

**What Needs To Be Built**

1. One canonical agent runtime.

2. Remove alternate benchmark scorers as primary paths.

3. Make Broca parsing mandatory for benchmark items.

4. Feed LLM logits into Tensegrity as evidence.

5. Move score fusion inside the belief update.

6. Add persistent agent state save/load.

7. Persist episodic, associative, epistemic, NGC, and causal arena state.

8. Treat benchmark correctness as post-action feedback.

9. Add online consolidation after each item.

10. Make final benchmark prediction come from the integrated posterior only.

The target is simple: one persistent predictive-causal-memory agent, with the LLM embedded as language input/output, learning continuously from consequences.