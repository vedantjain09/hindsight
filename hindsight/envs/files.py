from __future__ import annotations

import random

from ..core import Env, ToolError

WORDS = ["alpha", "beta", "gamma", "delta", "error", "ok", "temp", "debug", "trace", "cache"]


class FilesEnv(Env):
    name = "files"
    TOOLS = {
        "ls": "ls(dir): list files directly under a directory",
        "read": "read(path): file contents",
        "grep": "grep(pattern, dir): files under dir whose contents contain pattern",
        "mv": "mv(src, dst): move a file (overwrites dst if it exists)",
        "rm": "rm(path): delete a file permanently",
    }

    def _build(self, r: random.Random) -> None:
        names = r.sample(list("abcdefghijklmn"), r.randint(7, 12))
        fs = {}
        for n in names:
            ext = r.choice([".log", ".log", ".txt", ".cfg"])
            fs[f"/app/{n}{ext}"] = " ".join(r.choice(WORDS) for _ in range(r.randint(2, 5)))
        fs["/archive/old.log"] = "previous archive"
        kind = "archive" if r.random() < 0.55 else "purge"
        plan, mistakes = [], [{"tool": "ls", "args": {"dir": "/archive"}}, {"tool": "read", "args": {"path": "/nope.txt"}}]
        self.state = {"fs": dict(fs)}
        exp = dict(fs)
        app = sorted(p for p in fs if p.startswith("/app/"))
        mistakes.append({"tool": "read", "args": {"path": r.choice(app)}})
        if kind == "archive":
            self.goal = "Move every .log file in /app into /archive, keeping the file names. Do not touch any other file."
            plan.append({"tool": "ls", "args": {"dir": "/app"}})
            for p in app:
                name = p.rsplit("/", 1)[1]
                if p.endswith(".log"):
                    plan.append({"tool": "mv", "args": {"src": p, "dst": f"/archive/{name}"}})
                    exp[f"/archive/{name}"] = exp.pop(p)
                else:
                    mistakes.append({"tool": "rm", "args": {"path": p}})
                    mistakes.append({"tool": "mv", "args": {"src": p, "dst": f"/archive/{name}"}})
            mistakes.append({"tool": "rm", "args": {"path": "/archive/old.log"}})
        else:
            word = r.choice(["error", "debug", "temp"])
            self.goal = f"Delete every file in /app whose contents contain the word '{word}'. Keep all other files."
            plan.append({"tool": "grep", "args": {"pattern": word, "dir": "/app"}})
            for p in app:
                if word in fs[p].split():
                    plan.append({"tool": "rm", "args": {"path": p}})
                    exp.pop(p)
                else:
                    mistakes.append({"tool": "rm", "args": {"path": p}})
        if not any(a["tool"] in ("mv", "rm") for a in plan):          # guarantee at least one real action
            p = app[0]
            plan.append({"tool": "rm", "args": {"path": p}})
            exp.pop(p, None)
            self.goal += f" Also delete {p}."
        self.expected_fs, self.plan, self.mistakes = exp, plan, mistakes

    def t_ls(self, dir):
        d = dir.rstrip("/") + "/"
        return sorted(p for p in self.state["fs"] if p.startswith(d) and "/" not in p[len(d):])

    def t_read(self, path):
        if path not in self.state["fs"]:
            raise ToolError("file not found")
        return self.state["fs"][path]

    def t_grep(self, pattern, dir):
        d = dir.rstrip("/") + "/"
        return sorted(p for p, c in self.state["fs"].items() if p.startswith(d) and pattern in c.split())

    def t_mv(self, src, dst):
        if src not in self.state["fs"]:
            raise ToolError("file not found")
        self.state["fs"][dst] = self.state["fs"].pop(src)
        return "moved"

    def t_rm(self, path):
        if path not in self.state["fs"]:
            raise ToolError("file not found")
        del self.state["fs"][path]
        return "deleted"

    def success(self) -> bool:
        return self.state["fs"] == self.expected_fs
