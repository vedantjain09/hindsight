import json
import random

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hindsight.api import create_app
from hindsight.datasets import build
from hindsight.policies import NoisyExpert
from hindsight.replay import locate_adaptive
from hindsight.rollout import collect
from hindsight.stats import mcnemar_exact, paired_bootstrap
from hindsight.store import TrajectoryStore

POL = NoisyExpert()


@pytest.fixture(scope="module")
def world():
    trs = collect("orders", list(range(60)), POL) + collect("files", list(range(60)), POL)
    st = TrajectoryStore()
    st.put(trs)
    rng = random.Random(0)
    atts = {t.id: locate_adaptive(t, POL, rng) for t in trs if not t.success}
    for tid, a in atts.items():
        st.put_curve(tid, [(t, v, n) for t, (v, n) in sorted(a.curve.items())], "adaptive")
    return st, trs, atts


def test_store_roundtrip(world):
    st, trs, _ = world
    t = trs[0]
    g = st.get(t.id)
    assert [s.action for s in g.steps] == [s.action for s in t.steps] and g.success == t.success
    assert g.snapshots[3] == t.snapshots[3]


def test_query_layer(world):
    st, trs, _ = world
    assert set(st.ids(success=False)) == {t.id for t in trs if not t.success}
    assert st.with_tool_errors("get_order")
    rates = st.tool_failure_rates()
    assert 0 <= min(rates.values()) and max(rates.values()) <= 1
    for tid, tool, run in st.retry_loops(2):
        assert run >= 2


def test_otel_export_shape(world):
    st, trs, _ = world
    spans = st.export_otel(trs[0].id)
    assert len(spans) == len(trs[0].steps) and spans[0]["parentSpanId"] == "" and spans[1]["parentSpanId"] == "0000"
    assert all(s["status"]["code"].startswith("STATUS_CODE_") for s in spans)


def test_dataset_builders(world):
    _, trs, atts = world
    d = build(trs, atts, POL, random.Random(0), k_alt=8, m=10)
    s = d["stats"]
    assert s["n_pairs"] > 5 and s["n_salvage_sft"] > s["n_outcome_sft"]
    for p in d["dpo"]:
        assert p["chosen"] != p["rejected"] and p["value_chosen"] >= p["value_rejected"] + 0.3
        json.loads(p["chosen"]); json.loads(p["rejected"])
    # salvage never contains the harmful decisive action at its decisive state
    harmful = {(p["prompt"], p["rejected"]) for p in d["dpo"]}
    assert not any((r["prompt"], r["completion"]) in harmful for r in d["salvage_sft"])


def test_api(world):
    st, trs, _ = world
    c = TestClient(create_app(st))
    assert c.get("/health").json()["ok"]
    failed = c.get("/trajectories", params={"success": "false"}).json()
    assert failed
    d = c.get(f"/trajectory/{failed[0]}").json()
    assert d["curve"] and d["steps"] and "predicted_decisive" in d
    assert c.get("/trajectory/nope").status_code == 404
    assert c.get("/stats").json()["n_trajectories"] == 120


def test_stats_helpers():
    rng = np.random.default_rng(0)
    base = (rng.random(300) < 0.4).astype(float)
    better = np.where(rng.random(300) < 0.25, 1.0, base)
    assert paired_bootstrap(base, better)["significant"]
    assert not paired_bootstrap(base, base.copy())["significant"]
    assert mcnemar_exact(np.array([1, 0, 0, 0], bool), np.array([1, 1, 1, 1], bool))["b_only"] == 3
