import pytest

from hindsight.core import render_prompt
from hindsight.envs import ENVS, make_env

SEEDS = range(120)


@pytest.mark.parametrize("name", list(ENVS))
def test_oracle_plan_solves_every_task_and_is_not_initially_doomed(name):
    for s in SEEDS:
        e = make_env(name, s)
        assert not e.doomed
        for a in e.plan:
            e._apply(a, flaky=False)
        assert e.success(), (name, s, e.goal)


@pytest.mark.parametrize("name", list(ENVS))
def test_snapshot_restore_is_exact_and_isolated(name):
    for s in range(20):
        e = make_env(name, s)
        snap = e.snapshot()
        before = repr(e.state)
        for a in e.plan:
            e.step(a)
        assert repr(snap["state"]) == before            # snapshot is a deep copy
        e.restore(snap)
        assert repr(e.state) == before


@pytest.mark.parametrize("name", list(ENVS))
def test_plan_is_idempotent(name):
    """Re-running the whole plan on a solved state must keep it solved (the doom check relies on this)."""
    for s in range(40):
        e = make_env(name, s)
        for _ in range(2):
            for a in e.plan:
                e._apply(a, flaky=False)
        assert e.success(), (name, s)


@pytest.mark.parametrize("name", list(ENVS))
def test_decisive_mistake_dooms_the_task(name):
    """For tasks with a consequential mistake, executing it flags `decisive` and the task can no longer succeed."""
    found = 0
    for s in range(150):
        e = make_env(name, s)
        for m in e.mistakes:
            c = make_env(name, s)
            _, err, tag = c.step(m)
            if tag == "decisive":
                found += 1
                for a in c.plan:
                    c._apply(a, flaky=False)
                assert not c.success()
                break
    assert found > 20


def test_harmless_lookups_never_doom():
    for name in ENVS:
        for s in range(60):
            e = make_env(name, s)
            for m in e.mistakes:
                if m["tool"] in ("get_order", "get_stock", "check_coupon", "ls", "read", "grep", "list_events", "find_slot"):
                    c = make_env(name, s)
                    assert c.step(m)[2] != "decisive", (name, s, m)


def test_horizons_are_long_enough_to_matter():
    lens = [len(make_env("orders", s).plan) for s in range(200)]
    assert max(lens) >= 15 and sum(lens) / len(lens) > 6


def test_prompt_never_contains_the_oracle_plan():
    e = make_env("orders", 3)
    p = render_prompt(e, [])
    assert e.goal in p and "ship" in p
    assert "O999" not in p and "plan" not in p.lower().split("goal:")[0]


def test_unknown_tool_and_bad_args_are_errors_not_crashes():
    e = make_env("files", 1)
    assert e.step({"tool": "rm_rf", "args": {}})[1] is True
    assert e.step({"tool": "ls", "args": {"nope": 1}})[1] is True
    assert e.step({"tool": "ls", "args": "not-a-dict"})[1] is True


def test_lm_policy_source_never_touches_oracle_fields():
    import inspect
    from hindsight import policies
    src = inspect.getsource(policies.LocalLM) + inspect.getsource(policies.parse_action)
    assert ".plan" not in src and ".mistakes" not in src
