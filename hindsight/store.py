"""Trajectory store (SQLite) with an OTEL-style span schema and a small query layer.

Each step becomes a span row; trajectories are the root. Snapshots are stored as JSON so replay works from the
store alone. `export_otel` emits OTLP-flavoured JSON spans (a subset of the OTLP/JSON shape, enough for ingestion
experiments; not a full OpenTelemetry exporter).
"""
from __future__ import annotations

import json
import sqlite3

from .core import Step, Trajectory

SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories(id TEXT PRIMARY KEY, env TEXT, seed INT, policy TEXT, goal TEXT, success INT, n_steps INT, snapshots TEXT);
CREATE TABLE IF NOT EXISTS spans(traj_id TEXT, step INT, tool TEXT, args TEXT, obs TEXT, error INT, tag TEXT, PRIMARY KEY(traj_id, step));
CREATE TABLE IF NOT EXISTS curves(traj_id TEXT, t INT, value REAL, n INT, method TEXT, PRIMARY KEY(traj_id, t, method));
CREATE INDEX IF NOT EXISTS ix_spans_tool ON spans(tool);
CREATE INDEX IF NOT EXISTS ix_traj_env ON trajectories(env, success);
"""


class TrajectoryStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)

    def put(self, trajs: list[Trajectory]) -> None:
        with self.db:
            for t in trajs:
                self.db.execute("INSERT OR REPLACE INTO trajectories VALUES(?,?,?,?,?,?,?,?)",
                                (t.id, t.env, t.seed, t.policy, t.goal, int(t.success), len(t.steps), json.dumps(t.snapshots)))
                self.db.execute("DELETE FROM spans WHERE traj_id=?", (t.id,))
                self.db.executemany("INSERT INTO spans VALUES(?,?,?,?,?,?,?)",
                                    [(t.id, i, s.action.get("tool"), json.dumps(s.action.get("args", {}), sort_keys=True),
                                      s.obs, int(s.error), s.tag) for i, s in enumerate(t.steps)])

    def get(self, tid: str) -> Trajectory:
        r = self.db.execute("SELECT id,env,seed,policy,goal,success,snapshots FROM trajectories WHERE id=?", (tid,)).fetchone()
        if r is None:
            raise KeyError(tid)
        steps = [Step({"tool": tool, "args": json.loads(args)}, obs, bool(err), tag, t=i)
                 for i, tool, args, obs, err, tag in self.db.execute(
                     "SELECT step,tool,args,obs,error,tag FROM spans WHERE traj_id=? ORDER BY step", (tid,))]
        return Trajectory(r[0], r[1], r[2], r[3], r[4], steps, json.loads(r[6]), bool(r[5]))

    def ids(self, env: str | None = None, success: bool | None = None, policy: str | None = None) -> list[str]:
        q, a = "SELECT id FROM trajectories WHERE 1=1", []
        for col, v in (("env", env), ("success", None if success is None else int(success)), ("policy", policy)):
            if v is not None:
                q += f" AND {col}=?"
                a.append(v)
        return [r[0] for r in self.db.execute(q + " ORDER BY id", a)]

    # ---- query layer ---------------------------------------------------------------------------------
    def with_tool_errors(self, tool: str | None = None) -> list[str]:
        q = "SELECT DISTINCT traj_id FROM spans WHERE error=1" + (" AND tool=?" if tool else "")
        return [r[0] for r in self.db.execute(q, [tool] if tool else [])]

    def retry_loops(self, min_run: int = 2) -> list[tuple[str, str, int]]:
        """(traj_id, tool, run_length): consecutive calls to the same tool where all but the last errored."""
        out = []
        rows = self.db.execute("SELECT traj_id,step,tool,error FROM spans ORDER BY traj_id,step").fetchall()
        i = 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0] and rows[j + 1][2] == rows[i][2] and rows[j][3] == 1:
                j += 1
            if j - i + 1 >= min_run:
                out.append((rows[i][0], rows[i][2], j - i + 1))
            i = j + 1
        return out

    def tool_failure_rates(self) -> dict[str, float]:
        return {t: e / n for t, n, e in self.db.execute("SELECT tool, COUNT(*), SUM(error) FROM spans GROUP BY tool")}

    def put_curve(self, tid: str, values: list[tuple[int, float, int]], method: str) -> None:
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO curves VALUES(?,?,?,?,?)", [(tid, t, v, n, method) for t, v, n in values])

    def curve(self, tid: str, method: str | None = None) -> list[dict]:
        q, a = "SELECT t,value,n,method FROM curves WHERE traj_id=?", [tid]
        if method:
            q += " AND method=?"
            a.append(method)
        return [dict(t=t, value=v, n=n, method=m) for t, v, n, m in self.db.execute(q + " ORDER BY t", a)]

    def export_otel(self, tid: str) -> list[dict]:
        t = self.get(tid)
        spans = []
        for i, s in enumerate(t.steps):
            spans.append({"traceId": tid, "spanId": f"{i:04d}", "parentSpanId": f"{i - 1:04d}" if i else "",
                          "name": f"tool.{s.action.get('tool')}", "kind": "SPAN_KIND_INTERNAL",
                          "status": {"code": "STATUS_CODE_ERROR" if s.error else "STATUS_CODE_OK"},
                          "attributes": {"agent.step": i, "tool.args": json.dumps(s.action.get("args", {}), sort_keys=True),
                                         "tool.result": s.obs[:500], "agent.goal": t.goal if i == 0 else None}})
        return spans
