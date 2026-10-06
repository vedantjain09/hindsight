from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .store import TrajectoryStore

WEB = Path(__file__).resolve().parent.parent / "web" / "index.html"


def create_app(store: TrajectoryStore) -> FastAPI:
    app = FastAPI(title="Hindsight")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/trajectories")
    def trajectories(env: str | None = None, success: bool | None = None, limit: int = 200):
        return store.ids(env, success)[:limit]

    @app.get("/trajectory/{tid}")
    def trajectory(tid: str):
        try:
            t = store.get(tid)
        except KeyError:
            raise HTTPException(404, "unknown trajectory")
        curve = store.curve(tid, "adaptive") or store.curve(tid)
        pred = None
        if curve:
            vals = sorted(curve, key=lambda c: c["t"])
            pred = next((c["t"] - 1 for c in vals if c["value"] < 0.25 and c["t"] > 0), None)
        return {"id": t.id, "env": t.env, "goal": t.goal, "success": t.success, "truth_decisive": t.decisive_step(),
                "predicted_decisive": pred, "curve": curve,
                "steps": [{"i": i, "action": s.action, "obs": s.obs[:300], "error": s.error, "tag": s.tag} for i, s in enumerate(t.steps)]}

    @app.get("/otel/{tid}")
    def otel(tid: str):
        return store.export_otel(tid)

    @app.get("/stats")
    def stats():
        return {"tool_failure_rates": store.tool_failure_rates(), "retry_loops": len(store.retry_loops(2)),
                "n_trajectories": len(store.ids()), "n_failed": len(store.ids(success=False))}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return WEB.read_text()

    return app
