"""Counterfactual replay: estimate the value V_t of each state along a failed trajectory by Monte Carlo rollouts
under a policy, and locate the DECISIVE step: the action after which the success probability collapses.

V_t = P(success | state before step t, policy). A failed run's final value is 0 by definition. Values stay high
until the decisive action, then drop. Harmless errors (flaky timeouts, redundant lookups) do not move V.

Two locators:
  exhaustive  estimate V_t at every step with a fixed rollout budget, take the largest drop.
  adaptive    binary search for the first t where V_t < tau. Each probe draws rollouts in chunks and stops as soon
              as a Wilson interval separates V_t from tau. Cost is O(log n) probes with early stopping.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .core import Trajectory
from .rollout import rollouts_from


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


@dataclass
class Attribution:
    step: int | None                      # decisive step index, None if unattributable
    cost: int                             # total rollouts used
    curve: dict[int, tuple[float, int]] = field(default_factory=dict)   # t -> (value estimate, n rollouts)
    note: str = ""


def _probe(traj, t, policy, rng, tau, chunk=4, m_max=40, first_action=None):
    k = n = 0
    while n < m_max:
        k += sum(rollouts_from(traj, t, policy, chunk, rng, first_action))
        n += chunk
        lo, hi = wilson(k, n)
        if hi < tau or lo > tau:
            break
    return k / n, n


def locate_adaptive(traj: Trajectory, policy, rng: random.Random, tau: float = 0.25, chunk: int = 4, m_max: int = 40,
                    rel: float = 0.4, m0: int = 24) -> Attribution:
    """The collapse threshold is relative to the run's starting value: tau_eff = clip(rel * V_0, 0.05, tau).
    (An absolute threshold misfires on long or hard tasks where V is legitimately low before the fatal step.)"""
    n_steps = len(traj.steps)
    curve, cost = {}, 0
    k0 = sum(rollouts_from(traj, 0, policy, m0, rng))
    v0, n0 = k0 / m0, m0
    curve[0] = (v0, n0)
    cost += n0
    if v0 < 0.08:
        return Attribution(None, cost, curve, "unattributable: low value from the start")
    tau = max(0.05, min(tau, rel * v0))
    lo, hi = 0, n_steps                    # V_lo >= tau, V_hi = 0 (failed run)
    curve[n_steps] = (0.0, 0)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        v, n = _probe(traj, mid, policy, rng, tau, chunk, m_max)
        curve[mid] = (v, n)
        cost += n
        if v >= tau:
            lo = mid
        else:
            hi = mid
    return Attribution(lo, cost, curve)


def locate_exhaustive(traj: Trajectory, policy, rng: random.Random, m: int = 24) -> Attribution:
    n_steps = len(traj.steps)
    curve, cost = {}, 0
    for t in range(n_steps):
        k = sum(rollouts_from(traj, t, policy, m, rng))
        curve[t] = (k / m, m)
        cost += m
    curve[n_steps] = (0.0, 0)
    best = max(range(n_steps), key=lambda t: curve[t][0] - curve[t + 1][0]) if n_steps else None
    return Attribution(best, cost, curve)


# ---- baselines that see only the logged trajectory (no replay) ---------------------------------------------
def first_error(traj: Trajectory) -> int | None:
    for i, s in enumerate(traj.steps):
        if s.error:
            return i
    return max(len(traj.steps) - 2, 0) if traj.steps else None


def last_error(traj: Trajectory) -> int | None:
    errs = [i for i, s in enumerate(traj.steps) if s.error]
    return errs[-1] if errs else (max(len(traj.steps) - 2, 0) if traj.steps else None)


def last_action(traj: Trajectory) -> int | None:
    fin = [i for i, s in enumerate(traj.steps) if s.action.get("tool") == "finish"]
    return (fin[0] - 1 if fin and fin[0] > 0 else len(traj.steps) - 1) if traj.steps else None


def random_step(traj: Trajectory, rng: random.Random) -> int | None:
    return rng.randrange(len(traj.steps)) if traj.steps else None
