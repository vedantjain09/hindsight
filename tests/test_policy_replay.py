import json
import random

import numpy as np
import pytest

from hindsight import replay
from hindsight.envs import make_env
from hindsight.policies import NoisyExpert, parse_action
from hindsight.replay import locate_adaptive, locate_exhaustive, wilson
from hindsight.rollout import collect, rollouts_from

POL = NoisyExpert()


def test_parse_action_variants():
    assert parse_action('{"tool": "ship", "args": {"order_id": "O1"}}')["tool"] == "ship"
    assert parse_action('Sure!\n```json\n{"tool": "ls", "args": {"dir": "/app"}}\n```')["args"] == {"dir": "/app"}
    assert parse_action('{"tool": "finish"}')["args"] == {}
    assert parse_action("I do not know")["tool"] == "invalid"
    assert parse_action('{"nottool": 1} then {"tool": "ls", "args": {}}')["tool"] == "ls"


def test_collect_is_deterministic():
    a = collect("files", list(range(10)), POL, rng_seed=3)
    b = collect("files", list(range(10)), POL, rng_seed=3)
    assert [[s.action for s in t.steps] for t in a] == [[s.action for s in t.steps] for t in b]


def test_snapshots_align_with_steps():
    for t in collect("orders", list(range(20)), POL):
        assert len(t.snapshots) == len(t.steps) + 1


def test_zero_noise_expert_always_succeeds():
    clean = NoisyExpert(0.0, 0.0)
    for env in ("orders", "files", "calendar"):
        trs = collect(env, list(range(60)), clean)
        assert all(t.success for t in trs)


def test_value_is_zero_after_doom_and_high_before():
    rng = random.Random(0)
    checked = 0
    for t in collect("orders", list(range(80)), POL):
        d = t.decisive_step()
        if t.success or d is None:
            continue
        assert sum(rollouts_from(t, d + 1, POL, 10, rng)) == 0
        assert sum(rollouts_from(t, d, POL, 20, rng)) >= 8
        checked += 1
        if checked == 6:
            break
    assert checked == 6


def test_counterfactual_intervention_can_rescue_a_run():
    rng = random.Random(1)
    n = 0
    for t in collect("orders", list(range(100)), POL):
        d = t.decisive_step()
        if t.success or d is None:
            continue
        env_plan = make_env(t.env, t.seed).plan
        done = {json.dumps(s.action, sort_keys=True) for s in t.steps[:d] if not s.error}
        nxt = next((a for a in env_plan if json.dumps(a, sort_keys=True) not in done), None)
        if nxt is None or nxt == t.steps[d].action:
            continue
        v = sum(rollouts_from(t, d, POL, 20, rng, first_action=nxt)) / 20
        assert v > 0.3
        n += 1
        if n == 5:
            break
    assert n >= 3


def test_adaptive_matches_ground_truth_often_and_beats_baselines():
    rng = random.Random(0)
    hits, base, tot = 0, 0, 0
    for env in ("orders", "files", "calendar"):
        for t in collect(env, list(range(40)), POL):
            if t.success or t.decisive_step() is None:
                continue
            tot += 1
            hits += locate_adaptive(t, POL, rng).step == t.decisive_step()
            base += replay.first_error(t) == t.decisive_step()
    assert tot > 30 and hits / tot > 0.8 and hits > 3 * base


def test_adaptive_is_cheaper_than_exhaustive_and_agrees():
    rng = random.Random(2)
    ca = ce = agree = n = 0
    for t in collect("orders", list(range(40)), POL):
        if t.success or t.decisive_step() is None or len(t.steps) < 12:
            continue
        a, e = locate_adaptive(t, POL, rng), locate_exhaustive(t, POL, rng, m=16)
        ca += a.cost
        ce += e.cost
        agree += a.step == e.step
        n += 1
        if n == 8:
            break
    assert n == 8 and ca < 0.5 * ce and agree >= 6


def test_harmless_error_before_the_fault_does_not_fool_replay():
    rng = random.Random(3)
    tested = 0
    for t in collect("calendar", list(range(150)), POL):
        d = t.decisive_step()
        if t.success or d is None or not any(s.error for s in t.steps[:d]):
            continue
        assert replay.first_error(t) < d                 # the decoy exists
        assert locate_adaptive(t, POL, rng).step == d    # replay is not fooled
        tested += 1
        if tested == 5:
            break
    assert tested >= 3


@pytest.mark.parametrize("k,n", [(0, 0), (0, 10), (10, 10), (5, 10)])
def test_wilson_bounds(k, n):
    lo, hi = wilson(k, n)
    assert 0 <= lo <= hi <= 1
    if n:
        assert lo <= k / n <= hi


def test_unattributable_when_value_low_from_start():
    class Hopeless(NoisyExpert):
        def act(self, env, history, rng):
            return {"tool": "finish", "args": {}}
    t = collect("orders", [1], POL)[0]
    t.success = False
    att = locate_adaptive(t, Hopeless(), random.Random(0))
    assert att.step is None
