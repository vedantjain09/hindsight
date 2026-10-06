"""Attribution benchmark: does counterfactual replay find the step that doomed a failed run?

Ground truth comes from the oracle doom check (core.py), which is policy-agnostic: the decisive step is the first
action after which re-running the oracle plan can no longer solve the task. Baselines see only the log.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict

import numpy as np

from . import replay
from .policies import NoisyExpert
from .rollout import collect

ENV_NAMES = ["orders", "files", "calendar"]


def _acc(pred, truth, tol=0):
    return float(pred is not None and abs(pred - truth) <= tol)


def run_benchmark(n_tasks: int = 200, seed: int = 0, eps_dec: float = 0.04, eps_harm: float = 0.08,
                  replay_policy=None, tau: float = 0.25, m_exh: int = 24, m_max: int = 40) -> dict:
    gen = NoisyExpert(eps_dec, eps_harm)
    rp = replay_policy or gen
    rng = random.Random(seed)
    res = defaultdict(lambda: defaultdict(list))
    cost = defaultdict(list)
    decoy = []
    n_fail = 0
    t0 = time.time()
    for env in ENV_NAMES:
        trajs = collect(env, list(range(seed * 10_000, seed * 10_000 + n_tasks)), gen, rng_seed=seed)
        for tr in trajs:
            if tr.success or tr.decisive_step() is None:
                continue
            truth = tr.decisive_step()
            n_fail += 1
            fe = replay.first_error(tr)
            decoy.append(float(fe is not None and fe < truth))
            preds = {"first_error": fe, "last_error": replay.last_error(tr), "last_action": replay.last_action(tr),
                     "random": replay.random_step(tr, rng)}
            a = replay.locate_adaptive(tr, rp, rng, tau=tau, m_max=m_max)
            e = replay.locate_exhaustive(tr, rp, rng, m=m_exh)
            preds["replay_adaptive"], preds["replay_exhaustive"] = a.step, e.step
            cost["replay_adaptive"].append(a.cost)
            cost["replay_exhaustive"].append(e.cost)
            for k, p in preds.items():
                res[k]["exact"].append(_acc(p, truth))
                res[k]["within1"].append(_acc(p, truth, 1))
                res[f"{k}"][f"env:{env}"].append(_acc(p, truth))
    out = {"n_failed_trajectories": n_fail, "n_tasks_per_env": n_tasks, "seed": seed, "eps_dec": eps_dec, "eps_harm": eps_harm,
           "tau": tau, "runtime_s": round(time.time() - t0, 1),
           "frac_failed_with_earlier_decoy_error": float(np.mean(decoy)), "methods": {}}
    for k, d in res.items():
        m = {"exact": float(np.mean(d["exact"])), "within1": float(np.mean(d["within1"]))}
        b = np.random.default_rng(0).choice(d["exact"], size=(2000, len(d["exact"]))).mean(1)
        m["exact_ci95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        for kk, v in d.items():
            if kk.startswith("env:"):
                m[kk] = float(np.mean(v))
        if k in cost:
            m["mean_rollouts"] = float(np.mean(cost[k]))
        out["methods"][k] = m
    ad, ex = out["methods"]["replay_adaptive"]["mean_rollouts"], out["methods"]["replay_exhaustive"]["mean_rollouts"]
    out["adaptive_rollout_savings"] = 1 - ad / ex
    return out
