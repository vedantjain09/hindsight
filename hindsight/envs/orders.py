from __future__ import annotations

import json
import random

from ..core import Env, ToolError


class OrdersEnv(Env):
    name = "orders"
    TOOLS = {
        "get_order": "get_order(order_id): order status, items [[sku, qty]], total, coupon",
        "get_stock": "get_stock(sku): units in stock",
        "check_coupon": "check_coupon(code): discount percent and minimum order total",
        "apply_coupon": "apply_coupon(order_id, code): apply a valid coupon to a pending order",
        "ship": "ship(order_id): ship a pending order (decrements stock)",
        "cancel": "cancel(order_id): cancel a pending order (cannot be undone)",
    }

    def _build(self, r: random.Random) -> None:
        skus = [f"SKU{i}" for i in range(8)]
        stock = {s: r.randint(0, 6) for s in skus}
        coupons = {"SAVE10": {"pct": 10, "min": 50}, "BIG20": {"pct": 20, "min": 120}}
        kind = "coupon" if r.random() < 0.3 else "batch"
        k = 1 if kind == "coupon" else r.choice([2, 2, 3, 4])
        orders = {}
        for i in range(k):
            items = [[s, r.randint(1, 3)] for s in r.sample(skus, r.randint(1, 3))]
            orders[f"O{100 + i}"] = {"status": "pending", "items": items, "coupon": None,
                                     "total": sum(q * r.randint(10, 45) for _, q in items)}
        self.state = {"orders": orders, "stock": stock, "coupons": coupons}
        self.expected: dict = {}
        plan: list[dict] = []
        mistakes: list[dict] = [{"tool": "get_order", "args": {"order_id": "O999"}},
                                {"tool": "check_coupon", "args": {"code": "BOGUS"}},
                                {"tool": "get_stock", "args": {"sku": r.choice(skus)}}]
        if kind == "batch":
            self.goal = ("Process orders " + ", ".join(orders) + ": ship each order only if every item is in stock "
                         "at the moment you process it (in the order listed); otherwise cancel it.")
            run = dict(stock)
            for oid, o in orders.items():
                plan.append({"tool": "get_order", "args": {"order_id": oid}})
                for s, _ in o["items"]:
                    plan.append({"tool": "get_stock", "args": {"sku": s}})
                ok = all(run[s] >= q for s, q in o["items"])
                if ok:
                    for s, q in o["items"]:
                        run[s] -= q
                    plan.append({"tool": "ship", "args": {"order_id": oid}})
                    mistakes.append({"tool": "cancel", "args": {"order_id": oid}})
                    self.expected[oid] = "shipped"
                else:
                    plan.append({"tool": "cancel", "args": {"order_id": oid}})
                    mistakes.append({"tool": "ship", "args": {"order_id": oid}})
                    self.expected[oid] = "cancelled"
            self.need_total = None
        else:
            oid, o = next(iter(orders.items()))
            for sk, q in o["items"]:
                stock[sk] = max(stock[sk], q + 1)       # coupon tasks are always shippable
            code = r.choice(list(coupons))
            valid = o["total"] >= coupons[code]["min"]
            self.goal = (f"Order {oid}: apply coupon {code} if it is valid for this order (check the minimum), "
                         f"then ship the order. Do not apply a coupon that is not valid.")
            plan = [{"tool": "get_order", "args": {"order_id": oid}}, {"tool": "check_coupon", "args": {"code": code}}]
            if valid:
                plan.append({"tool": "apply_coupon", "args": {"order_id": oid, "code": code}})
            else:
                mistakes.append({"tool": "apply_coupon", "args": {"order_id": oid, "code": code}})
            plan.append({"tool": "ship", "args": {"order_id": oid}})
            self.expected[oid] = "shipped"
            self.coupon_expect = {oid: code if valid else None}
            # shipping before the coupon is applied locks the order: the pending-only rule makes it unrecoverable
            if valid:
                mistakes.append({"tool": "ship", "args": {"order_id": oid}})
            mistakes.append({"tool": "cancel", "args": {"order_id": oid}})
        self.kind, self.plan, self.mistakes = kind, plan, mistakes

    # tools
    def _o(self, oid):
        if oid not in self.state["orders"]:
            raise ToolError(f"no such order {oid}")
        return self.state["orders"][oid]

    def t_get_order(self, order_id):
        return json.dumps(self._o(order_id))

    def t_get_stock(self, sku):
        if sku not in self.state["stock"]:
            raise ToolError(f"no such sku {sku}")
        return self.state["stock"][sku]

    def t_check_coupon(self, code):
        c = self.state["coupons"].get(code)
        if c is None:
            raise ToolError("unknown coupon")
        return json.dumps(c)

    def t_apply_coupon(self, order_id, code):
        o, c = self._o(order_id), self.state["coupons"].get(code)
        if c is None or o["status"] != "pending" or o["coupon"] or o["total"] < c["min"]:
            raise ToolError("coupon not applicable")
        o["total"] = o["total"] * (100 - c["pct"]) // 100
        o["coupon"] = code
        return "coupon applied"

    def t_ship(self, order_id):
        o = self._o(order_id)
        if o["status"] != "pending":
            raise ToolError(f"order is {o['status']}")
        for s, q in o["items"]:
            self.state["stock"][s] -= q        # silent oversell if stock is short
        o["status"] = "shipped"
        return "shipped"

    def t_cancel(self, order_id):
        o = self._o(order_id)
        if o["status"] != "pending":
            raise ToolError(f"order is {o['status']}")
        o["status"] = "cancelled"
        return "cancelled"

    def success(self) -> bool:
        st = self.state
        if any(v < 0 for v in st["stock"].values()):
            return False
        for oid, want in self.expected.items():
            if st["orders"][oid]["status"] != want:
                return False
        if self.kind == "coupon":
            return all(st["orders"][o]["coupon"] == c for o, c in self.coupon_expect.items())
        return True
