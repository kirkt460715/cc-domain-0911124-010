"""批次级核销演练。

合成同一客户、同一币种下的交叉争抢场景，对比逐笔贪心（受输入顺序支配）
与全局批次求解的差异，并演示同分替代、部分回款、弱证据边影响、人工锁定
重算以及确认前余额变化的整批拒绝。所有金额按币种最小单位守恒。
"""

from .service import Workbench
from .primitives import BookError


def _post(work, ident, kind, amount, reference, party="factory",
          currency="CNY", issued="2026-01-05", due=None):
    row = {
        "id": ident,
        "kind": kind,
        "party": party,
        "currency": currency,
        "amount": amount,
        "issued": issued,
        "reference": reference,
    }
    if kind == "invoice":
        row["due"] = due or "2026-01-31"
    work.ledger.post(row)


def _greedy_single(work, source, effective):
    """模拟旧的逐笔入口：只看当前回款，按最高分拿走唯一匹配项。"""
    report = work.matching.candidates(source, effective)
    if not report["candidates"]:
        return []
    remaining = report["available_minor"]
    taken = []
    for candidate in report["candidates"]:
        amount = min(remaining, candidate["amount_minor"])
        if amount <= 0:
            break
        taken.append((source, candidate["invoice"], amount))
        remaining -= amount
        if not remaining:
            break
    return taken


def run():
    findings = {}
    with Workbench() as work:
        work.ledger.register_party("factory", "合成仪器厂", ["Instrument Factory"])

        # 场景一：交叉争抢。p1/p2 同时到账，i1 是两张回款的共同候选，
        # 但 i1 的参考号只与 p1 匹配，i2 只与 p2 匹配。
        _post(work, "i1", "invoice", "100.00", "REF-1", due="2026-01-20")
        _post(work, "i2", "invoice", "100.00", "REF-2", due="2026-02-10")
        _post(work, "p1", "payment", "100.00", "REF-1")
        _post(work, "p2", "payment", "100.00", "REF-2")
        effective = "2026-02-01"

        # 逐笔贪心：按输入顺序先处理 p1，再处理 p2（此处结果恰好可行，
        # 但调换顺序会暴露顺序依赖——见场景二）。
        greedy_pairs = []
        greedy_balances = {}
        for source in ("p1", "p2"):
            greedy_pairs.extend(_greedy_single(work, source, effective))
        greedy_pairs = sorted(greedy_pairs)

        batch = work.clearing.propose(
            "contest", "factory", "CNY", effective,
            ["p1", "p2"], ["i1", "i2"],
        )
        global_pairs = sorted(
            (item["source"], item["invoice"], item["amount_minor"])
            for item in batch["report"]["allocations"]
        )
        findings["cross_contest"] = {
            "greedy_pairs": greedy_pairs,
            "global_pairs": global_pairs,
            "global_objective": batch["report"]["objective"],
            "order_independent": _order_independent(work, effective),
            "conservation": batch["report"]["conservation"],
        }

        # 场景二：同分替代。两张发票参考号、金额、到期完全相同，全局求解
        # 只能任选其一，必须显式标注歧义而不是静默决定。
        _post(work, "j1", "invoice", "80.00", "SAME", due="2026-01-25")
        _post(work, "j2", "invoice", "80.00", "SAME", due="2026-01-25")
        _post(work, "q1", "payment", "80.00", "SAME")
        tied = work.clearing.propose(
            "tied", "factory", "CNY", effective, ["q1"], ["j1", "j2"]
        )
        findings["equal_score_alternatives"] = {
            "ambiguous": tied["report"]["ambiguous"],
            "ambiguity_reasons": tied["report"]["ambiguity_reasons"],
            "chosen_edges": [item["edge"] for item in tied["report"]["allocations"]],
            "alternative_edges": [
                row["diff_edges"] for row in tied["report"]["alternatives"]
            ],
            "stable_digest": tied["report"]["solution_digest"],
        }

        # 场景三：部分回款与未分配款。回款大于发票开放额，允许部分核销并
        # 显式保留未分配款，金额按最小单位守恒。
        _post(work, "k1", "invoice", "30.00", "PART", due="2026-03-15")
        _post(work, "r1", "payment", "45.50", "PART")
        partial = work.clearing.propose(
            "partial", "factory", "CNY", effective, ["r1"], ["k1"]
        )
        findings["partial_payment"] = {
            "allocations": [
                (item["edge"], item["amount_minor"])
                for item in partial["report"]["allocations"]
            ],
            "unapplied": partial["report"]["unapplied"],
            "objective": partial["report"]["objective"],
        }

        # 场景四：弱证据边。参考号不匹配且发票未到期，唯一候选边没有任何
        # 证据加分，只是数值上可行；去掉它后目标恶化程度帮助财务识别这类
        # “只是数值上最优”的核销。
        _post(work, "m1", "invoice", "50.00", "WEAK", due="2026-03-20")
        _post(work, "s1", "payment", "20.00", "OTHER")
        weak = work.clearing.propose(
            "weak", "factory", "CNY", effective, ["s1"], ["m1"]
        )
        impact = work.clearing.edge_impact("weak", "s1", "m1")
        findings["weak_evidence"] = {
            "weak_edges": weak["report"]["weak_evidence_edges"],
            "edge_impact_deltas": impact["deltas"],
            "still_feasible_without_edge": impact["still_feasible"],
        }

        # 场景五：人工锁定后重算其余部分。
        _post(work, "n1", "invoice", "100.00", "LOCK", due="2026-01-18")
        _post(work, "t1", "payment", "100.00", "LOCK")
        work.clearing.propose(
            "locked", "factory", "CNY", effective, ["t1"], ["n1"]
        )
        work.clearing.lock("locked", "t1", "n1", "25.00", "treasurer", "客户电话确认先付25")
        recomputed = work.clearing.recompute("locked")
        findings["manual_lock"] = {
            "allocations": [
                {
                    "edge": item["edge"],
                    "amount_minor": item["amount_minor"],
                    "locked": item["locked"],
                }
                for item in recomputed["report"]["allocations"]
            ],
            "revision": recomputed["revision"],
        }

        # 场景六：确认前余额变化。另一笔分配先入账，批次建议随即过时，
        # 确认必须整批拒绝、不部分入账，并保留失败原因与原建议。
        _post(work, "v1", "invoice", "60.00", "STALE", due="2026-01-22")
        _post(work, "u1", "payment", "60.00", "STALE")
        work.clearing.propose(
            "stale", "factory", "CNY", effective, ["u1"], ["v1"]
        )
        work.ledger.allocate("outside", "u1", "v1", "12.00", effective, "manual entry")
        stale_code = None
        try:
            work.clearing.confirm("stale", "treasurer")
        except BookError as exc:
            stale_code = exc.code
        findings["stale_confirmation"] = {
            "rejected_code": stale_code,
            "batch_state_after_failure": work.clearing.get("stale")["state"],
            "failure_reasons": [
                row["reason"] for row in work.clearing.failures("stale")
            ],
            "outside_allocation_only": len(
                work.db.rows("SELECT * FROM allocations WHERE reason='manual entry'")
            ) == 1,
        }

        # 正常批次确认与重复确认幂等。
        confirmed = work.clearing.confirm("contest", "treasurer")
        repeat = work.clearing.confirm("contest", "treasurer")
        findings["confirmation"] = {
            "state": confirmed["state"],
            "idempotent": repeat["result"] == confirmed["result"],
            "allocation_count": len(work.db.rows("SELECT * FROM allocations")),
            "controls_ok": work.controls.require_valid()["ok"],
        }
    return {"status": "ok", "findings": findings, "offline": True}


def _order_independent(work, effective):
    """同一问题调换输入次序，方案摘要必须完全一致。"""
    first = work.clearing.propose(
        "ord-a", "factory", "CNY", effective, ["p1", "p2"], ["i1", "i2"]
    )
    second = work.clearing.propose(
        "ord-b", "factory", "CNY", effective, ["p2", "p1"], ["i2", "i1"]
    )
    return (
        first["report"]["solution_digest"]
        == second["report"]["solution_digest"]
    )
