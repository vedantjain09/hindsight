# Hindsight

Counterfactual credit assignment for long-horizon agent trajectories, and a pipeline that turns the result into
post-training data.

When an agent run of 10 to 20 tool calls fails, the outcome says *that* it failed, not *which step* ruined it.
Hindsight finds that step by replaying from each state with the agent's own policy and measuring how the probability
of eventual success changes. Harmless errors (flaky timeouts, redundant lookups) do not move that probability; the
decisive mistake collapses it. The located step then yields step-level training signal with no human labels:
preference pairs at the decisive state, and salvaged good steps from runs that failed.

```
stateful tool-use envs ──► trajectories ──► span store + query layer (OTEL-style)
 (orders, files, calendar;   (snapshot every     │
  serializable state,         step)              ▼
  flaky tools, verifiers)              counterfactual replay engine
                                       Monte Carlo value V_t per step
                                       adaptive binary search | exhaustive
                                                 │
                       ┌─────────────────────────┼──────────────────────────┐
                       ▼                         ▼                          ▼
              value-curve viewer        training data builders       attribution benchmark
              (FastAPI + SVG)           outcome SFT / step-salvage    (vs log-only baselines,
                                        SFT / step-level DPO          oracle ground truth)
                                                 │
                                       LoRA SFT / DPO / eval scripts (Colab)
```

## Ground truth without human labels

Each task ships an oracle plan (a list of idempotent tool calls that solves it). A run is **doomed** once re-running
the whole oracle plan from the current state no longer solves the task (flaky tools off). The **decisive step** is the
first action after which the run is doomed. This is policy-agnostic, so it labels scripted *and* LLM trajectories.
Environments include silent irreversible actions (a cancelled order, a deleted file, a double-booked slot) alongside
loud-but-harmless errors, so "first error" and "last error" heuristics are decoyed on purpose.

## What has been measured (and what has not)

Measured, offline, no GPU or API: attribution accuracy on 3 environments x 100 held-out task seeds each (fresh seed 1;
the replay rule was tuned on seed 0, see "Honest notes"). Policy is the scripted noisy expert (follows the oracle plan,
injects harmless and consequential mistakes). Ground truth is the oracle doom check.

| Condition (failed runs) | Replay, adaptive | Replay, exhaustive | First error | Last error | Random |
|---|---|---|---|---|---|
| Default noise (124) | **100%** | 98.4% | 5.6% | 5.6% | 4.0% |
| Harsher noise (188) | **93.6%** | 95.7% | 4.8% | 4.8% | 6.9% |
| Replay under a mismatched policy (124) | **89.5%** | 94.4% | 5.6% | 5.6% | 5.6% |
| Stricter collapse threshold (124) | **100%** | 99.2% | 5.6% | 5.6% | 9.7% |

Exact-step match. Adaptive search uses about 74 to 97 rollouts per failed run versus about 450 for exhaustive replay
(78% to 84% fewer). 16% to 18% of failed runs contain a harmless error *before* the fatal step, which is what fools the
log-only baselines. Full results with 95% bootstrap intervals: `results/benchmarks.json`. Reproduce:
`SEED=1 python scripts/run_benchmarks.py` (about 4 minutes).

Verified by smoke test only: `scripts/train_sft.py`, `train_dpo.py`, `eval_policy.py` and the Hugging Face policy ran
end to end on CPU against a tiny randomly initialised model (DPO loss starts at ln 2 and preference accuracy rises).
`tests/test_dpo.py` checks the loss against a NumPy reference.

**Not measured yet, and the central open question:**

- **No real LLM has been run.** Every number above uses the scripted policy, so it validates the replay machinery and
  the benchmark, not agent improvement. The Colab path (LocalLM policy, replay with batched generation) is written but
  untested on a GPU with a real model.
- **No post-training result.** Whether step-level DPO or step-salvage SFT beats outcome-only rejection sampling at an
  equal data budget is unmeasured. It may turn out small or null; the plan is to report whatever the paired bootstrap
  and McNemar tests say.
- Real LLM trajectories will be messier than the scripted ones (value curves will not drop cleanly to zero), so
  expect lower attribution accuracy there.

## Honest notes

- The benchmark is synthetic and favourable to replay by construction: fatal mistakes are silent, decoy errors are
  loud. It shows replay beats log-only heuristics *when that structure exists*, not that it generalises to production
  traces.
- My first replay rule used an absolute collapse threshold. On seed 0 it dropped to about 61% to 63% under harsher
  noise and policy mismatch because long or hard tasks have legitimately low value before the fatal step. I switched
  to a threshold relative to the run's starting value, then re-measured on fresh seed 1. The seed-0 results of the first
  rule are kept in `results/seed0_absolute_threshold/` for transparency.
- The oracle doom check approximates "unrecoverable". An LLM could in principle recover from states the oracle calls
  doomed, so for LLM runs treat replay as the measurement and the oracle label as a proxy.
- Adaptive search assumes value is roughly monotone around the collapse. Runs with two separate value drops will
  surface only the first.

## Quick start

```bash
pip install -e '.[dev]'
pytest                                       # 40 tests, ~10s
hindsight demo --n 80                        # collect, attribute, build datasets, write runs/demo.db
hindsight serve --db runs/demo.db            # value-curve viewer at http://127.0.0.1:8000
SEED=1 python scripts/run_benchmarks.py      # reproduce the benchmark table

# real-model path (Colab, free T4):  pip install -e '.[train]'
hindsight collect --policy lm --model Qwen/Qwen2.5-1.5B-Instruct --n 300 --db runs/lm.db
hindsight attribute --policy lm --db runs/lm.db
hindsight build-data --policy lm --db runs/lm.db --out-dir runs/data
python scripts/train_sft.py --data runs/data/outcome_sft.jsonl --out runs/lora_outcome --max-examples 1500
python scripts/train_sft.py --data runs/data/salvage_sft.jsonl --out runs/lora_salvage --max-examples 1500
python scripts/train_dpo.py --data runs/data/dpo.jsonl --out runs/lora_dpo
python scripts/eval_policy.py --adapter runs/lora_salvage --out runs/eval_salvage.json   # held-out seeds
hindsight compare runs/eval_outcome.json runs/eval_salvage.json                          # paired bootstrap + McNemar
```

Keep conditions at an equal data budget (`--max-examples`) so the comparison is about data quality, not quantity.

## Layout

| Path | Role |
|---|---|
| `hindsight/envs/` | orders, files, calendar environments: snapshot/restore, flaky tools, verifiers, oracle plans |
| `hindsight/core.py` | `Env` base, oracle doom check, step tagging, prompt rendering |
| `hindsight/policies.py`, `rollout.py` | noisy-expert and Hugging Face policies; batched lock-step rollouts and interventions |
| `hindsight/replay.py` | Monte Carlo value curves, adaptive and exhaustive locators, log-only baselines |
| `hindsight/store.py` | SQLite span store, query layer (error runs, retry loops, tool failure rates), OTEL-style export |
| `hindsight/datasets.py`, `train_utils.py` | outcome/salvage SFT and DPO builders; DPO loss |
| `hindsight/bench.py`, `stats.py` | attribution benchmark; paired bootstrap and exact McNemar |
| `hindsight/api.py`, `web/` | FastAPI and the value-curve viewer |
| `scripts/` | Colab LoRA SFT, DPO, held-out evaluation; benchmark runner |
