"""Episode collection and batched counterfactual rollouts."""
from __future__ import annotations

import random

from .core import Env, Step, Trajectory
from .envs import make_env
from .policies import act_batch


def _advance(envs, histories, snaps_list, policy, rngs, max_steps):
    """Run all episodes to completion in lock-step (so an LM can batch one generation per step)."""
    alive = [i for i, e in enumerate(envs) if not e.done and len(histories[i]) < max_steps]
    while alive:
        actions = act_batch(policy, [(envs[i], histories[i]) for i in alive], [rngs[i] for i in alive])
        nxt = []
        for i, a in zip(alive, actions):
            if snaps_list is not None:
                snaps_list[i].append(envs[i].snapshot())
            obs, err, tag = envs[i].step(a)
            histories[i].append(Step(a, obs, err, tag, t=len(histories[i])))
            if not envs[i].done and len(histories[i]) < max_steps:
                nxt.append(i)
        alive = nxt


def collect(env_name: str, seeds: list[int], policy, rng_seed: int = 0) -> list[Trajectory]:
    envs = [make_env(env_name, s) for s in seeds]
    rngs = [random.Random(rng_seed * 1_000_003 + s) for s in seeds]
    for e, r in zip(envs, rngs):
        e.reseed(r.randrange(1 << 30))
    hist = [[] for _ in envs]
    snaps = [[] for _ in envs]
    _advance(envs, hist, snaps, policy, rngs, max(e.max_steps for e in envs))
    out = []
    for e, h, sn, s in zip(envs, hist, snaps, seeds):
        sn.append(e.snapshot())
        out.append(Trajectory(id=f"{env_name}-{s}-{policy.name}", env=env_name, seed=s, policy=policy.name, goal=e.goal,
                              steps=h, snapshots=sn, success=e.success()))
    return out


def rollouts_from(traj: Trajectory, t: int, policy, m: int, rng: random.Random, first_action: dict | None = None) -> list[bool]:
    """Resume `m` independent episodes from the state BEFORE step t of `traj`, under `policy`.
    If first_action is given, step t is replaced by it (a counterfactual intervention)."""
    base = make_env(traj.env, traj.seed)
    envs, hist, rngs = [], [], []
    for _ in range(m):
        e = base.clone() if False else make_env(traj.env, traj.seed)
        e.restore(traj.snapshots[t])
        e.reseed(rng.randrange(1 << 30))
        envs.append(e)
        hist.append([Step(s.action, s.obs, s.error, s.tag, s.t) for s in traj.steps[:t]])
        rngs.append(random.Random(rng.randrange(1 << 30)))
    if first_action is not None:
        for e, h in zip(envs, hist):
            obs, err, tag = e.step(first_action)
            h.append(Step(first_action, obs, err, tag, t=len(h)))
    cap = max(e.max_steps for e in envs)
    _advance(envs, hist, None, policy, rngs, cap)
    return [e.success() for e in envs]
