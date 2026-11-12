"""批次级核销求解器。

纯函数、不触碰账本：输入快照（回款、开放发票、锁定项、权重），输出按
硬约束→证据收益→未分配金额→拆分成本排序的全局方案。所有金额均为币种
最小单位整数，全程仅用整数运算。

目标的精确语义（整数、可复算）：
- 硬约束由容量网络保证：同一客户/币种的候选边构成二分网络，源点→回款
  （容量=可用余额）、回款→发票（容量=边上限）、发票→汇点（容量=开放
  余额）。锁定项先从容量中扣减，因此锁定部分必然被满足，绝不跨客户或
  跨币种拼平。
- 第一阶段（多项式时间）：最小费用最大流优化基础证据（参考号、到期按
  流向的最小单位金额加权），并通过每条边“首单位激活弧 + 剩余单位弧”
  的固定费用计入拆分成本；费率放大 SPLIT_SCALE 倍后字典序严格为
  基础证据 → 已分配金额（未分配最少）→ 拆分笔数。
- 第二阶段（确定性后处理）：在不改变基础证据、总流量与拆分笔数的前提
  下，按固定弧序重路由流量，让尽可能多的发票由单条边全额覆盖，逐张
  发票累计 exact_open 奖励；无法在不牺牲主目标时获得的奖励不会被计入，
  避免“只是数值上最优但证据不足”被掩盖。
- 平局稳定：所有迭代次序按 id/费率确定性排序，最短增广路使用固定弧序
  Dijkstra（费用全部平移为非负整数）；同分替代方案通过逐条排除已选边
  重算来识别并显式标注歧义。
"""

from ..primitives import error, digest, integer


DEFAULT_WEIGHTS = {"reference": 100, "exact_open": 50, "due": 10}

MAX_SOURCES = 20
MAX_INVOICES = 40
MAX_EDGES = 160
MAX_ALTERNATIVES = 20

# 大于任何单边单位费率（reference + due 上限 1000+1000）。
RATE_CEILING = 10000
# 大于任何方案可能出现的拆分笔数，保证激活费用不会翻转证据费率的字典序。
SPLIT_SCALE = 1000


class MinCostFlow:
    """固定弧序 SPFA 最小费用最大流，结果只依赖网络构造次序。

    网络规模很小（至多约 360 条弧），SPFA 对残余网络中的负费用回退弧
    天然鲁棒，避免位势法在部分节点不可达时的更新陷阱。
    """

    def __init__(self):
        self.adj = {}
        self.capacity = []
        self.cost = []
        self.reverse = []
        self.ends = []

    def add_edge(self, upstream, downstream, capacity, cost):
        forward = len(self.capacity)
        self.capacity.extend((capacity, 0))
        self.cost.extend((cost, -cost))
        self.ends.extend(((upstream, downstream), (downstream, upstream)))
        self.reverse.extend((forward + 1, forward))
        self.adj.setdefault(upstream, []).append(forward)
        self.adj.setdefault(downstream, []).append(forward + 1)

    def solve(self, source, sink):
        flow = 0
        cost = 0
        while True:
            distance = {source: 0}
            parent = {source: -1}
            queue = [source]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                base = distance[node]
                for arc in self.adj[node]:
                    if self.capacity[arc] <= 0:
                        continue
                    target = self.ends[arc][1]
                    candidate = base + self.cost[arc]
                    if target not in distance or candidate < distance[target]:
                        distance[target] = candidate
                        parent[target] = arc
                        if target not in queue[head:]:
                            queue.append(target)
            if sink not in parent:
                return flow, cost
            bottleneck = None
            arc = parent[sink]
            while arc != -1:
                bottleneck = self.capacity[arc] if bottleneck is None else min(
                    bottleneck, self.capacity[arc]
                )
                arc = parent[self.ends[arc][0]]
            arc = parent[sink]
            while arc != -1:
                self.capacity[arc] -= bottleneck
                self.capacity[self.reverse[arc]] += bottleneck
                arc = parent[self.ends[arc][0]]
            flow += bottleneck
            cost += bottleneck * distance[sink]

    def arc_flow(self, arc):
        return self.capacity[self.reverse[arc]]


def build_edges(sources, invoices, effective, weights):
    """依据快照构造候选边，证据沿用参考号、可用余额与到期信息。"""
    edges = []
    for source in sorted(sources, key=lambda row: row["id"]):
        for invoice in sorted(invoices, key=lambda row: row["id"]):
            cap = min(source["available_minor"], invoice["open_minor"])
            if cap <= 0:
                continue
            reasons = []
            rate = 0
            if (
                source["reference"]
                and source["reference"].casefold() == invoice["reference"].casefold()
            ):
                rate += weights["reference"]
                reasons.append("exact_reference")
            if invoice["due"] <= effective:
                rate += weights["due"]
                reasons.append("due")
            edges.append(
                {
                    "key": source["id"] + "->" + invoice["id"],
                    "source": source["id"],
                    "invoice": invoice["id"],
                    "cap_minor": cap,
                    "rate": rate,
                    "reasons": reasons,
                }
            )
    return edges


def _base_flow(edges, source_cap, invoice_cap):
    """第一阶段：非负费用最小费用最大流，输出每条候选边的流量。"""
    network = MinCostFlow()
    sink = "T"
    for source, capacity in sorted(source_cap.items()):
        network.add_edge("S", "src:" + source, capacity, 0)
    for invoice, capacity in sorted(invoice_cap.items()):
        network.add_edge("inv:" + invoice, sink, capacity, 0)
    topology = []
    for edge in edges:
        cap = edge["cap_minor"]
        rate = edge["rate"]
        gate = "gate:" + edge["key"]
        first_arc = len(network.capacity)
        # 首单位承担一次激活费用（拆分成本）。
        network.add_edge(
            "src:" + edge["source"],
            gate,
            1,
            (RATE_CEILING - rate) * SPLIT_SCALE + 1,
        )
        rest_arc = len(network.capacity)
        network.add_edge(
            "src:" + edge["source"],
            gate,
            max(cap - 1, 0),
            (RATE_CEILING - rate) * SPLIT_SCALE,
        )
        join_arc = len(network.capacity)
        network.add_edge(gate, "inv:" + edge["invoice"], cap, 0)
        topology.append((edge, first_arc, rest_arc, join_arc))
    flow, _cost = network.solve("S", sink)
    amounts = {}
    for edge, first_arc, rest_arc, _join in topology:
        amount = network.arc_flow(first_arc) + network.arc_flow(rest_arc)
        amounts[edge["key"]] = amount
    return flow, amounts


def _consolidate(edges, source_cap, invoice_cap, initial):
    """第二阶段：固定基础证据与总流量，最小化实际使用的候选边数。

    在初始流已用边构成的子网络上，按“费率降、容量大、id 小”的固定
    优先级再跑一次最小费用流：优先级高的边费用更低，流量会尽量集中到
    少数边，从而在不改变基础证据（费率与总流量不变）的前提下消除
    1+9999 这类无意义拆分。结果只依赖确定性的弧序与费用。
    """
    active = {key for key, amount in initial.items() if amount > 0}
    if not active:
        return dict(initial)
    active_edges = [edge for edge in edges if edge["key"] in active]
    ranked = sorted(
        active_edges,
        key=lambda row: (-row["rate"], -row["cap_minor"], row["key"]),
    )
    priority = {row["key"]: index for index, row in enumerate(ranked)}
    network = MinCostFlow()
    sink = "T"
    for source, capacity in sorted(source_cap.items()):
        network.add_edge("S", "src:" + source, capacity, 0)
    for invoice, capacity in sorted(invoice_cap.items()):
        network.add_edge("inv:" + invoice, sink, capacity, 0)
    topology = []
    for edge in ranked:
        gate = "gate:" + edge["key"]
        # 首单位仍计一次激活费用，确保边数最少；优先级越低费用越高。
        first_arc = len(network.capacity)
        network.add_edge(
            "src:" + edge["source"], gate, 1, priority[edge["key"]] * 2 + 1
        )
        rest_arc = len(network.capacity)
        network.add_edge(
            "src:" + edge["source"],
            gate,
            max(edge["cap_minor"] - 1, 0),
            priority[edge["key"]] * 2,
        )
        join_arc = len(network.capacity)
        network.add_edge(gate, "inv:" + edge["invoice"], edge["cap_minor"], 0)
        topology.append((edge, first_arc, rest_arc, join_arc))
    flow, _cost = network.solve("S", sink)
    if flow != sum(initial.values()):
        # 子网络不应损失流量；若发生则退回初始流，保证安全。
        return dict(initial)
    amounts = {}
    for edge, first_arc, rest_arc, _join in topology:
        amounts[edge["key"]] = network.arc_flow(first_arc) + network.arc_flow(
            rest_arc
        )
    for edge in edges:
        amounts.setdefault(edge["key"], 0)
    return amounts


def _score(edges, invoice_cap, amounts, weights):
    """根据最终流量汇总逐边证据与全额覆盖奖励。"""
    used = []
    covered_count = {}
    for edge in edges:
        amount = amounts[edge["key"]]
        if amount <= 0:
            continue
        covered_count[edge["invoice"]] = covered_count.get(edge["invoice"], 0) + 1
    for edge in sorted(edges, key=lambda row: row["key"]):
        amount = amounts[edge["key"]]
        if amount <= 0:
            continue
        full_cover = (
            covered_count[edge["invoice"]] == 1
            and amount == invoice_cap[edge["invoice"]]
        )
        reasons = list(edge["reasons"])
        if full_cover:
            reasons.append("exact_open_amount")
        evidence = edge["rate"] * amount + (
            weights["exact_open"] if full_cover else 0
        )
        used.append(
            {
                "source": edge["source"],
                "invoice": edge["invoice"],
                "edge": edge["key"],
                "amount_minor": amount,
                "base_rate": edge["rate"],
                "evidence_score": evidence,
                "reasons": reasons,
            }
        )
    used.sort(key=lambda row: (row["source"], row["invoice"], row["edge"]))
    return used


def _optimize(edges, source_cap, invoice_cap, weights):
    flow, amounts = _base_flow(edges, source_cap, invoice_cap)
    amounts = _consolidate(edges, source_cap, invoice_cap, amounts)
    used = _score(edges, invoice_cap, amounts, weights)
    return flow, used


def solve(
    sources,
    invoices,
    effective,
    weights,
    locks=(),
    dropped=(),
    maximum_alternatives=5,
):
    """求批次全局最优核销方案，返回确定性的方案报告。"""
    integer(maximum_alternatives, 1, MAX_ALTERNATIVES)
    sources = sorted(sources, key=lambda row: row["id"])
    invoices = sorted(invoices, key=lambda row: row["id"])
    if len(sources) > MAX_SOURCES or len(invoices) > MAX_INVOICES:
        error("capacity", "batch clearing problem exceeds supported size")
    for key in weights:
        if key not in DEFAULT_WEIGHTS:
            error("input", "unknown evidence weight", weight=key)
        integer(weights[key], 0, 1000)

    source_cap = {row["id"]: row["available_minor"] for row in sources}
    invoice_cap = {row["id"]: row["open_minor"] for row in invoices}
    locked_rows = sorted(locks, key=lambda row: (row["source"], row["invoice"]))
    locked_pairs = set()
    for lock in locked_rows:
        if lock["source"] not in source_cap or lock["invoice"] not in invoice_cap:
            error("input", "lock references document outside batch")
        if lock["amount_minor"] <= 0:
            error("input", "lock amount must be positive")
        pair = (lock["source"], lock["invoice"])
        if pair in locked_pairs:
            error("input", "duplicate lock for same edge")
        locked_pairs.add(pair)
        source_cap[lock["source"]] -= lock["amount_minor"]
        invoice_cap[lock["invoice"]] -= lock["amount_minor"]
        if source_cap[lock["source"]] < 0 or invoice_cap[lock["invoice"]] < 0:
            error("input", "locked amount exceeds available balance")

    all_edges = build_edges(sources, invoices, effective, weights)
    all_edge_keys = {edge["key"] for edge in all_edges}
    dropped_keys = {row["source"] + "->" + row["invoice"] for row in dropped}
    locked_keys = {row["source"] + "->" + row["invoice"] for row in locked_rows}
    for key in dropped_keys:
        if key not in all_edge_keys and key not in locked_keys:
            error("input", "dropped edge is not part of the batch", edge=key)
    edges = [edge for edge in all_edges if edge["key"] not in dropped_keys]
    if len(edges) > MAX_EDGES:
        error(
            "capacity",
            "candidate edge limit exceeded; narrow the batch scope",
            edges=len(edges),
            limit=MAX_EDGES,
        )

    flow, chosen = _optimize(edges, source_cap, invoice_cap, weights)
    objective_tuple = (
        sum(row["evidence_score"] for row in chosen),
        flow,
        -len(chosen),
    )

    # 同分替代：逐条排除已选边后重算，目标三元组完全相同即为平局。
    alternatives = []
    truncated = False
    for row in sorted(chosen, key=lambda item: item["edge"]):
        alt_flow, alt_used = _optimize(
            [edge for edge in edges if edge["key"] != row["edge"]],
            source_cap,
            invoice_cap,
            weights,
        )
        alt_tuple = (
            sum(item["evidence_score"] for item in alt_used),
            alt_flow,
            -len(alt_used),
        )
        if alt_tuple == objective_tuple:
            if len(alternatives) >= maximum_alternatives:
                truncated = True
                continue
            chosen_keys = {item["edge"] for item in chosen}
            alternatives.append(
                {
                    "flow_digest": digest(alt_used),
                    "objective": {
                        "evidence_score": alt_tuple[0],
                        "allocated_minor": alt_flow,
                        "unapplied_minor": sum(source_cap.values()) - alt_flow,
                        "splits": len(alt_used),
                    },
                    "diff_edges": sorted(
                        chosen_keys ^ {item["edge"] for item in alt_used}
                    ),
                    "removed_edge": row["edge"],
                }
            )

    locked_allocation = [
        {
            "source": row["source"],
            "invoice": row["invoice"],
            "edge": row["source"] + "->" + row["invoice"],
            "amount_minor": row["amount_minor"],
            "base_rate": None,
            "evidence_score": None,
            "reasons": ["manually_locked"],
            "locked": True,
        }
        for row in locked_rows
    ]
    allocated_by_source = {row["id"]: 0 for row in sources}
    for row in locked_rows:
        allocated_by_source[row["source"]] += row["amount_minor"]
    for row in chosen:
        allocated_by_source[row["source"]] += row["amount_minor"]
    unapplied = [
        {
            "source": source["id"],
            "amount_minor": source["available_minor"]
            - allocated_by_source[source["id"]],
        }
        for source in sources
    ]

    total_sources = sum(row["available_minor"] for row in sources)
    total_invoices = sum(row["open_minor"] for row in invoices)
    total_locked = sum(row["amount_minor"] for row in locked_rows)
    total_allocated = total_locked + flow
    total_unapplied = total_sources - total_allocated
    if total_allocated + total_unapplied != total_sources:
        error("integrity", "batch conservation failed on source side")
    if total_allocated > total_invoices:
        error("integrity", "batch conservation failed on invoice side")

    edge_rates = {edge["key"]: edge["rate"] for edge in edges}
    weak_edges = [row["edge"] for row in chosen if edge_rates.get(row["edge"], 0) == 0]
    ambiguous = bool(alternatives) or truncated
    return {
        "effective": effective,
        "weights": dict(weights),
        "edges": [
            {
                "edge": edge["key"],
                "source": edge["source"],
                "invoice": edge["invoice"],
                "cap_minor": edge["cap_minor"],
                "base_rate": edge["rate"],
                "reasons": edge["reasons"],
            }
            for edge in sorted(edges, key=lambda row: row["key"])
        ],
        "locked": locked_allocation,
        "allocations": locked_allocation
        + [dict(row, locked=False) for row in chosen],
        "unapplied": unapplied,
        "objective": {
            "evidence_score": sum(row["evidence_score"] for row in chosen),
            "allocated_minor": total_allocated,
            "unapplied_minor": total_unapplied,
            "splits": len(locked_rows) + len(chosen),
        },
        "alternatives": alternatives,
        "ambiguous": ambiguous,
        "ambiguity_reasons": sorted(
            (["equal_score_alternative"] if alternatives else [])
            + (["alternatives_truncated"] if truncated else [])
        ),
        "weak_evidence_edges": weak_edges,
        "conservation": {
            "source_available_minor": total_sources,
            "invoice_open_minor": total_invoices,
            "allocated_minor": total_allocated,
            "unapplied_minor": total_unapplied,
            "balanced": total_allocated + total_unapplied == total_sources,
        },
        "solution_digest": digest(
            {
                "allocations": locked_allocation + chosen,
                "unapplied": unapplied,
                "weights": dict(weights),
            }
        ),
    }
