from __future__ import annotations

import json
import random

from ..core import Env, ToolError

USERS = ["alice", "bob", "carol"]


def _free(events, users, day, hrs):
    for s in range(9, 18 - hrs + 1):
        if all(not (e["day"] == day and set(e["who"]) & {u} and e["start"] < s + hrs and s < e["end"])
               for e in events.values() for u in users):
            return s
    return None


class CalendarEnv(Env):
    name = "calendar"
    TOOLS = {
        "list_events": "list_events(user, day): events for a user on a day (hours are 9-17)",
        "find_slot": "find_slot(users, day, hours): earliest start hour when all users are free",
        "create_event": "create_event(title, day, start, end, who): create an event",
        "delete_event": "delete_event(event_id): permanently delete an event",
        "move_event": "move_event(event_id, day, start): move an event keeping its duration",
    }

    def _build(self, r: random.Random) -> None:
        events = {}
        for i in range(r.randint(4, 8)):
            s = r.randint(9, 15)
            events[f"E{i}"] = {"title": r.choice(["standup", "1:1", "review", "lunch"]), "day": r.randint(1, 3),
                               "start": s, "end": s + r.randint(1, 2), "who": r.sample(USERS, r.randint(1, 2))}
        self.state = {"events": events, "next": len(events)}
        day = r.randint(1, 3)
        kind = "schedule" if r.random() < 0.6 else "move"
        mistakes = [{"tool": "list_events", "args": {"user": "carol", "day": 9}},
                    {"tool": "find_slot", "args": {"users": ["alice"], "day": 1, "hours": 1}}]
        if kind == "schedule":
            k = r.choice([1, 2, 3, 4])
            cur = {kk: dict(v) for kk, v in events.items()}
            self.plan, titles = [], []
            for i in range(k):
                u, d, s = None, None, None
                for _ in range(20):
                    u, d = r.sample(USERS, 2), r.randint(1, 3)
                    s = _free(cur, u, d, 1)
                    if s is not None:
                        break
                title = f"Sync{i}"
                titles.append((title, u, d, s))
                cur[f"N{i}"] = {"title": title, "day": d, "start": s, "end": s + 1, "who": u}
                self.plan += [{"tool": "list_events", "args": {"user": x, "day": d}} for x in u] + [
                    {"tool": "find_slot", "args": {"users": u, "day": d, "hours": 1}},
                    {"tool": "create_event", "args": {"title": title, "day": d, "start": s, "end": s + 1, "who": u}}]
                bad = s + 1 if s + 1 < 17 else 9
                mistakes.append({"tool": "create_event", "args": {"title": title, "day": d, "start": bad, "end": bad + 1, "who": u}})
            self.goal = ("Schedule these 1-hour meetings, one at a time in this order, each at the earliest hour (9-17) when both "
                         "people are free, taking earlier new meetings into account: " +
                         "; ".join(f"'{t}' on day {d} for {u[0]} and {u[1]}" for t, u, d, _ in titles) +
                         ". Do not change any existing event.")
            self.expected_events = [dict(v) for v in cur.values()]
            for kk, e in list(events.items())[:3]:
                mistakes.append({"tool": "delete_event", "args": {"event_id": kk}})
        else:
            eid = r.choice(list(events))
            e = events[eid]
            self.goal = (f"Event {eid} ('{e['title']}') on day {e['day']} must move to day {day}. Move it to the earliest "
                         f"hour on day {day} when all of its attendees are free. Do not delete events.")
            dur = e["end"] - e["start"]
            others = {k: v for k, v in events.items() if k != eid}
            s = _free(others, e["who"], day, dur)
            if s is None:
                for k in [k for k, v in others.items() if v["day"] == day]:
                    del events[k]
                others = {k: v for k, v in events.items() if k != eid}
                s = _free(others, e["who"], day, dur)
            self.plan = [{"tool": "list_events", "args": {"user": e["who"][0], "day": day}},
                         {"tool": "find_slot", "args": {"users": e["who"], "day": day, "hours": dur}},
                         {"tool": "move_event", "args": {"event_id": eid, "day": day, "start": s}}]
            exp = [dict(v) for k, v in events.items() if k != eid]
            exp.append({**e, "day": day, "start": s, "end": s + dur})
            self.expected_events = exp
            mistakes.append({"tool": "delete_event", "args": {"event_id": eid}})
            for k in list(others)[:2]:
                mistakes.append({"tool": "delete_event", "args": {"event_id": k}})
        self.state = {"events": events, "next": len(events) + 100}
        self.mistakes = mistakes

    def t_list_events(self, user, day):
        return json.dumps({k: e for k, e in self.state["events"].items() if user in e["who"] and e["day"] == day}, sort_keys=True)

    def t_find_slot(self, users, day, hours):
        s = _free(self.state["events"], users, day, hours)
        if s is None:
            raise ToolError("no common slot")
        return s

    def t_create_event(self, title, day, start, end, who):
        ev = {"title": title, "day": day, "start": start, "end": end, "who": list(who)}
        if any(e == ev for e in self.state["events"].values()):
            raise ToolError("duplicate event")
        self.state["events"][f"E{self.state['next']}"] = ev
        self.state["next"] += 1
        return "created"

    def t_delete_event(self, event_id):
        if event_id not in self.state["events"]:
            raise ToolError("no such event")
        del self.state["events"][event_id]
        return "deleted"

    def t_move_event(self, event_id, day, start):
        e = self.state["events"].get(event_id)
        if e is None:
            raise ToolError("no such event")
        d = e["end"] - e["start"]
        e.update(day=day, start=start, end=start + d)
        return "moved"

    def success(self) -> bool:
        key = lambda e: (e["title"], e["day"], e["start"], e["end"], tuple(sorted(e["who"])))
        return sorted(map(key, self.state["events"].values())) == sorted(map(key, self.expected_events))
