"""Core types and the Env base class.

Design notes
------------
* Environments are small stateful tool-use worlds with a JSON-serializable state, so a trajectory can be
  snapshotted and restored at ANY step. That is what makes counterfactual replay possible.
* Each task carries an ORACLE plan (a list of tool calls that solves it from the initial state). The oracle is
  used for exactly two things: (1) a scripted noisy-expert policy for free experiments, and (2) ground-truth
  labelling. It is never shown to a model policy.
* Ground truth for "which step ruined the run" is the first step after which the task is DOOMED: re-running the
  full oracle plan from the resulting state (flaky tools off) no longer solves the task. Plan actions are
  idempotent, so this is well defined from any state.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field


class ToolError(Exception):
    pass


def akey(a: dict) -> str:
    return json.dumps(a, sort_keys=True)


FINISH = {"tool": "finish", "args": {}}


@dataclass
class Step:
    action: dict
    obs: str
    error: bool = False
    tag: str = "ok"          # ground truth, benchmark-only: ok | harmless_error | decisive | after_doom
    t: int = 0


@dataclass
class Trajectory:
    id: str
    env: str
    seed: int
    policy: str
    goal: str
    steps: list[Step]
    snapshots: list[dict]    # snapshots[i] = state BEFORE step i; snapshots[len(steps)] = final state
    success: bool
    meta: dict = field(default_factory=dict)

    def decisive_step(self) -> int | None:
        """Ground-truth index of the step that doomed the run (None if never doomed)."""
        for i, s in enumerate(self.steps):
            if s.tag == "decisive":
                return i
        return None


class Env:
    name = "base"
    TOOLS: dict[str, str] = {}
    flaky_p = 0.04

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed * 7919 + 1)
        self.state: dict = {}
        self.goal = ""
        self.plan: list[dict] = []
        self.mistakes: list[dict] = []     # candidate wrong/harmless actions for the scripted noisy policy
        self.done = False
        self._build(random.Random(seed))
        self.doomed = False

    # --- to override -------------------------------------------------------------------------------
    def _build(self, r: random.Random) -> None:
        raise NotImplementedError

    def success(self) -> bool:
        raise NotImplementedError

    # --- snapshots ----------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"state": copy.deepcopy(self.state), "done": self.done}

    def restore(self, snap: dict) -> None:
        self.state = copy.deepcopy(snap["state"])
        self.done = snap["done"]
        self.doomed = self._plan_fails_from_here()

    def reseed(self, r: int) -> None:
        self.rng = random.Random(r)

    def clone(self) -> "Env":
        e = type(self)(self.seed)
        e.restore(self.snapshot())
        return e

    # --- stepping ----------------------------------------------------------------------------------
    def _apply(self, action: dict, flaky: bool) -> tuple[str, bool]:
        tool, args = action.get("tool"), action.get("args", {})
        if tool == "finish":
            self.done = True
            return "done", False
        fn = getattr(self, "t_" + str(tool), None)
        if fn is None or not isinstance(args, dict):
            return f"error: unknown tool {tool}", True
        if flaky and self.rng.random() < self.flaky_p:
            return "error: timeout, try again", True
        try:
            return str(fn(**args)), False
        except ToolError as e:
            return f"error: {e}", True
        except TypeError:
            return "error: bad arguments", True

    def step(self, action: dict) -> tuple[str, bool, str]:
        was_doomed = self.doomed
        obs, err = self._apply(action, flaky=True)
        self.doomed = self._plan_fails_from_here()
        if was_doomed:
            tag = "after_doom"
        elif self.doomed:
            tag = "decisive"
        else:
            tag = "harmless_error" if err else "ok"
        return obs, err, tag

    def _plan_fails_from_here(self) -> bool:
        """Oracle doom check: replay the whole (idempotent) oracle plan on a copy with flakiness off."""
        e = type(self)(self.seed)
        e.state = copy.deepcopy(self.state)
        for a in self.plan:
            e._apply(a, flaky=False)
        return not e.success()

    @property
    def max_steps(self) -> int:
        return 2 * len(self.plan) + 4

    def tool_doc(self) -> str:
        return "\n".join(f"- {n}: {d}" for n, d in self.TOOLS.items()) + "\n- finish(): call when the task is complete"


def render_prompt(env: Env, history: list[Step]) -> str:
    h = "\n".join(f"{i + 1}. {json.dumps(s.action)} -> {s.obs}" for i, s in enumerate(history)) or "(none yet)"
    return (f"You are a tool-using agent. Complete the goal by calling tools one at a time.\n\nGoal: {env.goal}\n\n"
            f"Tools:\n{env.tool_doc()}\n\nReply with exactly one JSON object: "
            f'{{"tool": <name>, "args": {{...}}}}.\n\nHistory:\n{h}\n\nNext action:')
