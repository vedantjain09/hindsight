"""Evaluate a base or adapter model on HELD-OUT task seeds; writes {env:seed: 0/1}.
  python scripts/eval_policy.py --adapter runs/lora_dpo --out runs/eval_dpo.json
Compare two conditions with:  python -m hindsight.cli compare runs/eval_a.json runs/eval_b.json"""
import argparse, json

p = argparse.ArgumentParser()
p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct"); p.add_argument("--adapter", default=None)
p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=150)
p.add_argument("--seed0", type=int, default=900_000, help="held-out seeds, disjoint from training seeds")
p.add_argument("--temperature", type=float, default=0.0)
a = p.parse_args()

from hindsight.policies import LocalLM
from hindsight.rollout import collect

pol = LocalLM(a.base, a.adapter, temperature=a.temperature)
res = {}
for env in ["orders", "files", "calendar"]:
    for tr in collect(env, list(range(a.seed0, a.seed0 + a.n)), pol):
        res[f"{env}:{tr.seed}"] = int(tr.success)
    print(env, sum(v for k, v in res.items() if k.startswith(env)) / a.n, flush=True)
json.dump(res, open(a.out, "w"))
