"""Reproduce results/benchmarks.json (offline, no GPU, no API)."""
import json, sys
sys.path.insert(0, ".")
from hindsight.bench import run_benchmark
from hindsight.policies import NoisyExpert

runs = {
    "default": dict(),
    "harsher_noise": dict(eps_dec=0.08, eps_harm=0.15),
    "mismatched_replay_policy": dict(replay_policy=NoisyExpert(0.08, 0.15)),
    "stricter_tau_0.4": dict(tau=0.4),
}
import os
SEED = int(os.environ.get("SEED", "1"))
only = sys.argv[1:] or list(runs)
for k in only:
    r = run_benchmark(n_tasks=100, seed=SEED, **runs[k])
    json.dump(r, open(f"results/bench_{k}.json", "w"), indent=1)
out = {k: json.load(open(f"results/bench_{k}.json")) for k in runs if os.path.exists(f"results/bench_{k}.json")}
json.dump(out, open("results/benchmarks.json", "w"), indent=1)
for k, r in out.items():
    m = r["methods"]
    print(k, "| failed:", r["n_failed_trajectories"], "| decoy-first-error frac:", round(r["frac_failed_with_earlier_decoy_error"], 2),
          "| adaptive", round(m["replay_adaptive"]["exact"], 3), "exhaustive", round(m["replay_exhaustive"]["exact"], 3),
          "first_error", round(m["first_error"]["exact"], 3), "random", round(m["random"]["exact"], 3),
          "| rollouts", round(m["replay_adaptive"]["mean_rollouts"]), "vs", round(m["replay_exhaustive"]["mean_rollouts"]),
          "| savings", round(r["adaptive_rollout_savings"], 2))
