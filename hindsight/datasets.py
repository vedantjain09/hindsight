"""Turn attributed trajectories into post-training data.

outcome_sft   every step of every SUCCESSFUL trajectory (the standard rejection-sampling baseline)
salvage_sft   outcome_sft  +  the pre-decisive steps of FAILED trajectories (replay shows value stayed high there,
              so those actions were fine)  +  the corrected action at the decisive state
dpo           (prompt, chosen, rejected) at each failed run's decisive state: `rejected` is what the agent did,
              `chosen` is the best counterfactual alternative by Monte Carlo value. Same state, different action,
              so the preference is about that one decision.
"""
from __future__ import annotations

import json
import random

from .core import Step, Trajectory, render_prompt
from .envs import make_env
from .policies import act_batch
from .replay import Attribution
from .rollout import rollouts_from


def _row(env, steps, i):
    return {"prompt": render_prompt(env, steps[:i]), "completion": json.dumps(steps[i].action)}


def sample_alternatives(traj: Trajectory, t: int, policy, rng: random.Random, k: int = 8) -> list[dict]:
    envs = []
    for _ in range(k):
        e = make_env(traj.env, traj.seed)
        e.restore(traj.snapshots[t])
        envs.append(e)
    hist = [traj.steps[:t] for _ in range(k)]
    acts = act_batch(policy, list(zip(envs, hist)), [random.Random(rng.randrange(1 << 30)) for _ in range(k)])
    uniq, seen = [], set()
    for a in acts:
        key = json.dumps(a, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    return uniq


def build(trajs: list[Trajectory], attributions: dict[str, Attribution], policy, rng: random.Random,
          k_alt: int = 8, m: int = 12, margin: float = 0.3) -> dict:
    outcome, salvage, dpo = [], [], []
    stats = {"n_traj": len(trajs), "n_success": 0, "n_failed": 0, "n_attributed": 0, "n_pairs": 0, "n_no_good_alt": 0}
    for tr in trajs:
        env = make_env(tr.env, tr.seed)
        if tr.success:
            stats["n_success"] += 1
            rows = [_row(env, tr.steps, i) for i in range(len(tr.steps))]
            outcome += rows
            salvage += rows
            continue
        stats["n_failed"] += 1
        att = attributions.get(tr.id)
        if att is None or att.step is None:
            continue
        stats["n_attributed"] += 1
        d = att.step
        salvage += [_row(env, tr.steps, i) for i in range(d)]
        v_actual = att.curve.get(d + 1, (0.0, 0))[0]
        best, best_v = None, -1.0
        for alt in sample_alternatives(tr, d, policy, rng, k_alt):
            if alt == tr.steps[d].action:
                continue
            v = sum(rollouts_from(tr, d, policy, m, rng, first_action=alt)) / m
            if v > best_v:
                best, best_v = alt, v
        if best is not None and best_v >= v_actual + margin:
            prompt = render_prompt(env, tr.steps[:d])
            dpo.append({"prompt": prompt, "chosen": json.dumps(best), "rejected": json.dumps(tr.steps[d].action),
                        "value_chosen": best_v, "value_rejected": v_actual, "traj": tr.id, "step": d})
            salvage.append({"prompt": prompt, "completion": json.dumps(best)})
            stats["n_pairs"] += 1
        else:
            stats["n_no_good_alt"] += 1
    stats.update(n_outcome_sft=len(outcome), n_salvage_sft=len(salvage), n_dpo=len(dpo))
    return {"outcome_sft": outcome, "salvage_sft": salvage, "dpo": dpo, "stats": stats}
