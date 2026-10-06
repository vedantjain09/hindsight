# Design notes

**Value of a state.** V_t = P(success | state before step t, policy). A failed run has V = 0 at its end. Along a run,
V rises slowly as work completes and collapses at the decisive action. Replay under the *same policy that produced the
run* answers "could this agent have recovered from here?", which is the quantity credit assignment needs.

**Why snapshots.** Environment state is plain JSON, so any step can be restored and re-rolled with fresh randomness.
Interventions (replace the action at step t, then continue) give counterfactual branches from one identical state, which
is what makes the preference pairs clean: same state, different action, different outcome.

**Adaptive search.** Binary search over steps for the first t with V_t below a collapse threshold, with Wilson-interval
early stopping per probe. The threshold is relative to V_0 so long or hard tasks are not misread.

**Batching.** Rollouts advance in lock-step, so an LLM policy generates one batched forward per step across all
concurrent rollouts instead of one call per rollout.

**Doom check as ground truth.** Plan actions are idempotent, so "re-run the oracle plan from this state" is well defined
from any state. It labels decisive steps for any policy, including LLM trajectories, without hand annotation.
