from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from .bench import ENV_NAMES, run_benchmark
from .datasets import build
from .policies import NoisyExpert
from .replay import locate_adaptive
from .rollout import collect
from .stats import mcnemar_exact, paired_bootstrap
from .store import TrajectoryStore


def _policy(a):
    if getattr(a, "policy", "scripted") == "lm":
        from .policies import LocalLM
        return LocalLM(a.model, a.adapter, temperature=a.temperature)
    return NoisyExpert(a.eps_dec, a.eps_harm)


def cmd_collect(a):
    st = TrajectoryStore(a.db)
    pol = _policy(a)
    for env in a.envs:
        trs = collect(env, list(range(a.seed0, a.seed0 + a.n)), pol, rng_seed=a.seed0)
        st.put(trs)
        print(f"{env}: {len(trs)} trajectories, success {np.mean([t.success for t in trs]):.2f}")


def _attribute(st, pol, tau, seed, ids=None):
    rng = random.Random(seed)
    out = {}
    for tid in ids or st.ids(success=False):
        tr = st.get(tid)
        att = locate_adaptive(tr, pol, rng, tau=tau)
        st.put_curve(tid, [(t, v, n) for t, (v, n) in sorted(att.curve.items())], "adaptive")
        out[tid] = att
    return out


def cmd_attribute(a):
    st = TrajectoryStore(a.db)
    atts = _attribute(st, _policy(a), a.tau, a.seed)
    hits = tot = cost = 0
    for tid, att in atts.items():
        truth = st.get(tid).decisive_step()
        cost += att.cost
        if truth is not None:
            tot += 1
            hits += att.step == truth
    print(f"{len(atts)} failed trajectories attributed, mean rollouts {cost / max(len(atts), 1):.1f}"
          + (f", exact match vs oracle ground truth {hits}/{tot}" if tot else ""))


def cmd_bench(a):
    r = run_benchmark(a.n, a.seed, a.eps_dec, a.eps_harm, tau=a.tau)
    json.dump(r, open(a.out, "w"), indent=1)
    for k, m in r["methods"].items():
        print(f"{k:20s} exact {m['exact']:.3f}  within1 {m['within1']:.3f}" + (f"  rollouts {m['mean_rollouts']:.0f}" if "mean_rollouts" in m else ""))
    print(f"adaptive rollout savings vs exhaustive: {100 * r['adaptive_rollout_savings']:.0f}%")


def cmd_build_data(a):
    st = TrajectoryStore(a.db)
    pol = _policy(a)
    atts = _attribute(st, pol, a.tau, a.seed)
    trs = [st.get(t) for t in st.ids(policy=pol.name)] if a.policy_filter else [st.get(t) for t in st.ids()]
    d = build(trs, atts, pol, random.Random(a.seed), k_alt=a.k_alt, m=a.m)
    os.makedirs(a.out_dir, exist_ok=True)
    for k in ("outcome_sft", "salvage_sft", "dpo"):
        with open(os.path.join(a.out_dir, f"{k}.jsonl"), "w") as f:
            for r in d[k]:
                f.write(json.dumps(r) + "\n")
    print(json.dumps(d["stats"], indent=1))


def cmd_compare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    ks = sorted(set(A) & set(B))
    va, vb = np.array([A[k] for k in ks], float), np.array([B[k] for k in ks], float)
    print(json.dumps({"n": len(ks), "pass_a": va.mean(), "pass_b": vb.mean(), "paired_bootstrap": paired_bootstrap(va, vb),
                      "mcnemar": mcnemar_exact(va, vb)}, indent=1, default=float))


def cmd_serve(a):
    import uvicorn
    from .api import create_app
    uvicorn.run(create_app(TrajectoryStore(a.db)), host="127.0.0.1", port=a.port)


def cmd_demo(a):
    os.makedirs(a.dir, exist_ok=True)
    db = os.path.join(a.dir, "demo.db")
    if os.path.exists(db):
        os.remove(db)
    st = TrajectoryStore(db)
    pol = NoisyExpert()
    for env in ENV_NAMES:
        st.put(collect(env, list(range(a.n)), pol))
    atts = _attribute(st, pol, 0.25, 0)
    trs = [st.get(t) for t in st.ids()]
    d = build(trs, atts, pol, random.Random(0))
    print(json.dumps(d["stats"], indent=1))
    print(f"serve with: hindsight serve --db {db}")


def main(argv=None):
    p = argparse.ArgumentParser("hindsight")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, db=True):
        if db:
            s.add_argument("--db", default="runs/hindsight.db")
        s.add_argument("--policy", choices=["scripted", "lm"], default="scripted")
        s.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
        s.add_argument("--adapter", default=None)
        s.add_argument("--temperature", type=float, default=0.7)
        s.add_argument("--eps-dec", type=float, default=0.04)
        s.add_argument("--eps-harm", type=float, default=0.08)
        s.add_argument("--tau", type=float, default=0.25)
        s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("collect"); common(s); s.add_argument("--envs", nargs="+", default=ENV_NAMES)
    s.add_argument("--n", type=int, default=200); s.add_argument("--seed0", type=int, default=0); s.set_defaults(f=cmd_collect)
    s = sub.add_parser("attribute"); common(s); s.set_defaults(f=cmd_attribute)
    s = sub.add_parser("bench"); common(s, db=False); s.add_argument("--n", type=int, default=200)
    s.add_argument("--out", default="results/benchmark.json"); s.set_defaults(f=cmd_bench)
    s = sub.add_parser("build-data"); common(s); s.add_argument("--out-dir", default="runs/data")
    s.add_argument("--k-alt", type=int, default=8); s.add_argument("--m", type=int, default=12)
    s.add_argument("--policy-filter", action="store_true"); s.set_defaults(f=cmd_build_data)
    s = sub.add_parser("compare"); s.add_argument("a"); s.add_argument("b"); s.set_defaults(f=cmd_compare)
    s = sub.add_parser("serve"); s.add_argument("--db", default="runs/hindsight.db"); s.add_argument("--port", type=int, default=8000); s.set_defaults(f=cmd_serve)
    s = sub.add_parser("demo"); s.add_argument("--dir", default="runs"); s.add_argument("--n", type=int, default=80); s.set_defaults(f=cmd_demo)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
