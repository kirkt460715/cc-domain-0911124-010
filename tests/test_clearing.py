import unittest

from receivables.service import Workbench
from receivables.primitives import BookError, money
from receivables.matching.clearing_solver import (
    solve,
    DEFAULT_WEIGHTS,
    MinCostFlow,
)


class ClearingSolverCase(unittest.TestCase):
    def source(self, ident, amount=10000, reference="R"):
        return {"id": ident, "reference": reference, "available_minor": amount}

    def invoice(self, ident, amount=10000, reference="R", due="2026-01-31"):
        return {"id": ident, "reference": reference, "due": due, "open_minor": amount}

    def test_flow_network_is_deterministic_and_bounded(self):
        network = MinCostFlow()
        network.add_edge("S", "a", 10, 0)
        network.add_edge("S", "b", 10, 0)
        network.add_edge("a", "T", 10, 1)
        network.add_edge("b", "T", 10, 2)
        flow, cost = network.solve("S", "T")
        self.assertEqual(flow, 20)
        self.assertEqual(cost, 30)
        self.assertEqual(network.arc_flow(0), 10)

    def test_cross_contest_prefers_global_reference_pairing(self):
        # 两张回款共享一张高分发票时，全局求解把参考号边同时保住。
        sources = [self.source("p1", reference="R1"), self.source("p2", reference="R2")]
        invoices = [
            self.invoice("i1", reference="R1"),
            self.invoice("i2", reference="R2", due="2026-02-15"),
        ]
        report = solve(sources, invoices, "2026-02-01", dict(DEFAULT_WEIGHTS))
        pairs = {(a["source"], a["invoice"]) for a in report["allocations"]}
        self.assertEqual(pairs, {("p1", "i1"), ("p2", "i2")})
        self.assertEqual(report["objective"]["unapplied_minor"], 0)
        self.assertTrue(report["conservation"]["balanced"])
        self.assertFalse(report["ambiguous"])

    def test_result_does_not_depend_on_input_order(self):
        sources = [self.source("p1", reference="R1"), self.source("p2", reference="R2")]
        invoices = [
            self.invoice("i1", reference="R1"),
            self.invoice("i2", reference="R2", due="2026-02-15"),
        ]
        first = solve(sources, invoices, "2026-02-01", dict(DEFAULT_WEIGHTS))
        second = solve(
            list(reversed(sources)),
            list(reversed(invoices)),
            "2026-02-01",
            dict(DEFAULT_WEIGHTS),
        )
        self.assertEqual(first["solution_digest"], second["solution_digest"])
        self.assertEqual(first["objective"], second["objective"])

    def test_equal_score_alternatives_are_reported_not_silently_picked(self):
        report = solve(
            [self.source("p1")],
            [self.invoice("i1"), self.invoice("i2")],
            "2026-02-01",
            dict(DEFAULT_WEIGHTS),
        )
        self.assertTrue(report["ambiguous"])
        self.assertEqual(report["ambiguity_reasons"], ["equal_score_alternative"])
        self.assertEqual(len(report["alternatives"]), 1)
        self.assertEqual(
            report["alternatives"][0]["diff_edges"], ["p1->i1", "p1->i2"]
        )
        # 重复运行的首选方案摘要稳定。
        again = solve(
            [self.source("p1")],
            [self.invoice("i1"), self.invoice("i2")],
            "2026-02-01",
            dict(DEFAULT_WEIGHTS),
        )
        self.assertEqual(
            report["allocations"], again["allocations"]
        )

    def test_partial_allocation_keeps_unapplied_and_conserves(self):
        report = solve(
            [self.source("p1", amount=4550, reference="X")],
            [self.invoice("i1", amount=3000, reference="Y", due="2026-03-15")],
            "2026-02-01",
            dict(DEFAULT_WEIGHTS),
        )
        self.assertEqual(report["allocations"][0]["amount_minor"], 3000)
        self.assertEqual(report["unapplied"], [{"source": "p1", "amount_minor": 1550}])
        self.assertEqual(
            report["objective"]["allocated_minor"]
            + report["objective"]["unapplied_minor"],
            4550,
        )

    def test_weak_evidence_edge_has_zero_base_score(self):
        report = solve(
            [self.source("p1", amount=2000, reference="OTHER")],
            [self.invoice("i1", amount=5000, reference="WEAK", due="2026-03-20")],
            "2026-02-01",
            dict(DEFAULT_WEIGHTS),
        )
        self.assertEqual(report["weak_evidence_edges"], ["p1->i1"])
        self.assertEqual(report["allocations"][0]["base_rate"], 0)
        self.assertEqual(report["allocations"][0]["evidence_score"], 0)

    def test_dropped_edge_changes_the_solution(self):
        sources = [self.source("p1", reference="R1")]
        invoices = [self.invoice("i1", reference="R1")]
        full = solve(sources, invoices, "2026-02-01", dict(DEFAULT_WEIGHTS))
        self.assertEqual(full["objective"]["allocated_minor"], 10000)
        dropped = solve(
            sources,
            invoices,
            "2026-02-01",
            dict(DEFAULT_WEIGHTS),
            dropped=[{"source": "p1", "invoice": "i1"}],
        )
        self.assertEqual(dropped["allocations"], [])
        self.assertEqual(dropped["objective"]["unapplied_minor"], 10000)

    def test_locks_are_satisfied_first_and_remain_distinct(self):
        report = solve(
            [self.source("p1", amount=6000)],
            [self.invoice("i1")],
            "2026-02-01",
            dict(DEFAULT_WEIGHTS),
            locks=[{"source": "p1", "invoice": "i1", "amount_minor": 2000}],
        )
        self.assertEqual(
            [(a["edge"], a["amount_minor"], a["locked"]) for a in report["allocations"]],
            [("p1->i1", 2000, True), ("p1->i1", 4000, False)],
        )
        self.assertEqual(report["objective"]["allocated_minor"], 6000)

    def test_lock_over_balance_is_rejected(self):
        with self.assertRaises(BookError):
            solve(
                [self.source("p1", amount=1000)],
                [self.invoice("i1")],
                "2026-02-01",
                dict(DEFAULT_WEIGHTS),
                locks=[{"source": "p1", "invoice": "i1", "amount_minor": 2000}],
            )

    def test_weights_change_objective_order(self):
        # 关闭参考号权重后，参考号匹配不再比普通到期匹配更优。
        sources = [self.source("p1", reference="R1")]
        invoices = [
            self.invoice("i1", reference="R1"),
            self.invoice("i2", reference="ZZ", due="2026-01-15"),
        ]
        weighted = solve(sources, invoices, "2026-02-01", dict(DEFAULT_WEIGHTS))
        self.assertEqual(weighted["allocations"][0]["invoice"], "i1")
        flat = solve(
            sources,
            invoices,
            "2026-02-01",
            {"reference": 0, "exact_open": 0, "due": 0},
        )
        self.assertEqual(flat["objective"]["evidence_score"], 0)

    def test_problem_size_limits_are_enforced(self):
        with self.assertRaises(BookError) as caught:
            solve(
                [self.source("p1")],
                [self.invoice(f"i{i}") for i in range(41)],
                "2026-02-01",
                dict(DEFAULT_WEIGHTS),
            )
        self.assertEqual(caught.exception.code, "capacity")

    def test_no_candidate_edge_still_balances(self):
        # 无可用边时保留全部未分配款且仍然守恒。
        report = solve(
            [self.source("p1", amount=10000, reference="N/A")],
            [],
            "2026-01-10",
            dict(DEFAULT_WEIGHTS),
        )
        self.assertEqual(report["allocations"], [])
        self.assertEqual(report["objective"]["unapplied_minor"], 10000)
        self.assertTrue(report["conservation"]["balanced"])

    def test_zero_base_score_edge_is_flagged_but_kept_when_nothing_else(self):
        # 无参考号、未到期：基础分为零，仅靠全额核销奖励入选，必须显式标弱证据。
        report = solve(
            [self.source("p1", reference="NONE")],
            [self.invoice("i1", reference="OTHER", due="2026-03-01")],
            "2026-01-10",
            dict(DEFAULT_WEIGHTS),
        )
        self.assertEqual(len(report["allocations"]), 1)
        self.assertEqual(report["allocations"][0]["reasons"], ["exact_open_amount"])
        self.assertEqual(report["weak_evidence_edges"], ["p1->i1"])
        self.assertTrue(report["conservation"]["balanced"])


class BatchClearingCase(unittest.TestCase):
    def setUp(self):
        self.w = Workbench()
        self.addCleanup(self.w.close)
        self.w.ledger.register_party("a", "Alpha", ["A Ltd"])
        self.w.ledger.register_party("b", "Beta")

    def post(self, ident, kind, amount, reference, party="a", currency="CNY",
             due="2026-01-31", issued="2026-01-05"):
        row = dict(id=ident, kind=kind, party=party, currency=currency,
                   amount=amount, issued=issued, reference=reference)
        if kind == "invoice":
            row["due"] = due
        self.w.ledger.post(row)

    def contest(self, batch_id="b1", effective="2026-02-01"):
        self.post("i1", "invoice", "100", "R1", due="2026-01-20")
        self.post("i2", "invoice", "100", "R2", due="2026-02-15")
        self.post("p1", "payment", "100", "R1")
        self.post("p2", "payment", "100", "R2")
        return self.w.clearing.propose(
            batch_id, "a", "CNY", effective, ["p1", "p2"], ["i1", "i2"]
        )

    def test_propose_is_read_only_and_idempotent(self):
        before = self.w.db.verify_audit()
        batch = self.contest()
        self.assertEqual(batch["state"], "proposed")
        self.assertEqual(batch["revision"], 1)
        self.assertEqual(self.w.db.rows("SELECT * FROM allocations"), [])
        again = self.contest()
        self.assertEqual(again["digest"], batch["digest"])
        self.assertEqual(self.w.ledger.balance("p1"), 10000)
        self.assertTrue(before["entries"] < self.w.db.verify_audit()["entries"])

    def test_proposal_identity_change_is_rejected(self):
        self.contest()
        with self.assertRaises(BookError) as caught:
            self.w.clearing.propose(
                "b1", "a", "CNY", "2026-02-02", ["p1", "p2"], ["i1", "i2"]
            )
        self.assertEqual(caught.exception.code, "conflict")

    def test_confirm_applies_everything_in_one_transaction(self):
        self.contest()
        result = self.w.clearing.confirm("b1", "treasurer")
        self.assertEqual(result["state"], "confirmed")
        self.assertEqual(len(result["result"]["allocations"]), 2)
        for ident in ("i1", "i2", "p1", "p2"):
            self.assertEqual(self.w.ledger.balance(ident), 0)
        self.assertTrue(self.w.controls.require_valid()["ok"])

    def test_repeat_confirmation_does_not_double_post(self):
        self.contest()
        first = self.w.clearing.confirm("b1", "treasurer")
        second = self.w.clearing.confirm("b1", "treasurer")
        self.assertEqual(first["result"], second["result"])
        self.assertEqual(len(self.w.db.rows("SELECT * FROM allocations")), 2)

    def test_stale_balance_rejects_whole_batch_and_keeps_advice(self):
        self.post("i1", "invoice", "60", "R3")
        self.post("p3", "payment", "60", "R3")
        self.w.clearing.propose("b2", "a", "CNY", "2026-02-01", ["p3"], ["i1"])
        self.w.ledger.allocate("outside", "p3", "i1", "12", "2026-02-02")
        with self.assertRaises(BookError) as caught:
            self.w.clearing.confirm("b2", "treasurer")
        self.assertEqual(caught.exception.code, "conflict")
        self.assertEqual(self.w.clearing.get("b2")["state"], "proposed")
        failures = self.w.clearing.failures("b2")
        self.assertEqual([row["reason"] for row in failures], ["conflict"])
        self.assertEqual(len(self.w.db.rows("SELECT * FROM allocations")), 1)

    def test_dispute_change_since_proposal_rejects_confirmation(self):
        self.post("i1", "invoice", "40", "R4")
        self.post("p4", "payment", "40", "R4")
        self.w.clearing.propose("b3", "a", "CNY", "2026-02-01", ["p4"], ["i1"])
        self.w.disputes.open("d", "i1", "5", "2026-02-03", "quality")
        with self.assertRaises(BookError) as caught:
            self.w.clearing.confirm("b3", "treasurer")
        self.assertEqual(caught.exception.code, "conflict")
        self.assertEqual(self.w.clearing.failures("b3")[0]["reason"], "conflict")

    def test_closed_period_rejects_batch_confirmation(self):
        self.contest()
        self.w.periods.create("feb", "2026-02-01", "2026-03-01")
        preview = self.w.periods.preview("feb")
        self.w.periods.close("feb", preview["digest"], "controller")
        with self.assertRaises(BookError) as caught:
            self.w.clearing.confirm("b1", "treasurer")
        self.assertEqual(caught.exception.code, "period")
        self.assertEqual(self.w.clearing.get("b1")["state"], "proposed")
        self.assertEqual(self.w.db.rows("SELECT * FROM allocations"), [])

    def test_cross_party_and_currency_are_never_mixed(self):
        self.post("ia", "invoice", "10", "R", party="a")
        self.post("pb", "payment", "10", "R", party="b")
        with self.assertRaises(BookError):
            self.w.clearing.propose("x", "a", "CNY", "2026-02-01", ["pb"], ["ia"])
        self.post("iu", "invoice", "10", "R", currency="USD")
        self.post("pu", "payment", "10", "R", currency="USD")
        with self.assertRaises(BookError):
            self.w.clearing.propose("y", "a", "CNY", "2026-02-01", ["pu"], ["iu"])

    def test_weight_versions_are_optimistic_and_reusable(self):
        profile = self.w.clearing.weight_save(
            "w1", {"reference": 200, "exact_open": 5, "due": 1}
        )
        self.assertEqual(profile["version"], 1)
        with self.assertRaises(BookError):
            self.w.clearing.weight_save(
                "w1", {"reference": 1, "exact_open": 1, "due": 1}
            )
        second = self.w.clearing.weight_save(
            "w1", {"reference": 1, "exact_open": 1, "due": 1}, expected_version=1
        )
        self.assertEqual(second["version"], 2)
        self.assertEqual(
            self.w.clearing.weight_get("w1")["weights"],
            {"reference": 1, "exact_open": 1, "due": 1},
        )

    def test_lock_then_recompute_fixes_manual_part(self):
        self.post("i1", "invoice", "100", "R9")
        self.post("p1", "payment", "60", "R9")
        self.w.clearing.propose("lk", "a", "CNY", "2026-02-01", ["p1"], ["i1"])
        self.w.clearing.lock("lk", "p1", "i1", "20", "treasurer", "电话确认")
        recomputed = self.w.clearing.recompute("lk")
        self.assertEqual(recomputed["revision"], 2)
        self.assertEqual(
            [(a["edge"], a["amount_minor"], a["locked"])
             for a in recomputed["report"]["allocations"]],
            [("p1->i1", 2000, True), ("p1->i1", 4000, False)],
        )
        confirmed = self.w.clearing.confirm("lk", "treasurer")
        self.assertEqual(confirmed["state"], "confirmed")
        self.assertEqual(self.w.ledger.balance("p1"), 0)
        self.assertEqual(len(self.w.clearing.revisions("lk")), 2)

    def test_unlock_restores_fully_open_solution(self):
        self.post("i1", "invoice", "100", "R9")
        self.post("p1", "payment", "60", "R9")
        self.w.clearing.propose("lk", "a", "CNY", "2026-02-01", ["p1"], ["i1"])
        self.w.clearing.lock("lk", "p1", "i1", "20", "treasurer", "电话确认")
        self.w.clearing.unlock("lk", "p1", "i1", "客户改主意")
        rec = self.w.clearing.recompute("lk")
        self.assertFalse(any(a["locked"] for a in rec["report"]["allocations"]))

    def test_confirmation_revalidates_cumulative_availability(self):
        # 同一发票同时含锁定与建议两笔分配，确认时按累计可用额重验，不误判。
        self.post("i1", "invoice", "100", "R9")
        self.post("p1", "payment", "60", "R9")
        self.w.clearing.propose("lk", "a", "CNY", "2026-02-01", ["p1"], ["i1"])
        self.w.clearing.lock("lk", "p1", "i1", "20", "treasurer", "电话确认")
        rec = self.w.clearing.recompute("lk")
        self.assertEqual(len(rec["report"]["allocations"]), 2)
        self.w.clearing.confirm("lk", "treasurer")
        self.assertEqual(self.w.ledger.balance("i1"), 4000)
        self.assertEqual(self.w.ledger.balance("p1"), 0)
        self.assertTrue(self.w.controls.require_valid()["ok"])

    def test_edge_impact_quantifies_removal(self):
        batch = self.contest()
        impact = self.w.clearing.edge_impact("b1", "p2", "i2")
        self.assertEqual(impact["objective_before"], batch["report"]["objective"])
        # 容量仍可由其他边承接，未分配不变，但证据收益明确下降。
        self.assertEqual(impact["deltas"]["unapplied_change_minor"], 0)
        self.assertLess(impact["deltas"]["evidence_score_change"], 0)
        self.assertTrue(impact["still_feasible"])
        with self.assertRaises(BookError):
            self.w.clearing.edge_impact("b1", "p1", "missing")

    def test_edge_impact_shows_unapplied_when_no_alternative_capacity(self):
        self.post("i1", "invoice", "60", "R3")
        self.post("p3", "payment", "60", "R3")
        self.w.clearing.propose("b2", "a", "CNY", "2026-02-01", ["p3"], ["i1"])
        impact = self.w.clearing.edge_impact("b2", "p3", "i1")
        self.assertEqual(impact["deltas"]["unapplied_change_minor"], 6000)
        self.assertFalse(impact["alternative_digest"] == self.w.clearing.get("b2")["report"]["solution_digest"])

    def test_recompute_with_dropped_edge_preserves_revision_history(self):
        self.contest()
        rec = self.w.clearing.recompute(
            "b1", dropped=[{"source": "p1", "invoice": "i1"}]
        )
        edges = {a["edge"] for a in rec["report"]["allocations"]}
        self.assertNotIn("p1->i1", edges)
        self.assertEqual(rec["revision"], 2)
        self.assertEqual(
            [row["revision"] for row in self.w.clearing.revisions("b1")], [1, 2]
        )

    def test_minor_unit_conservation_with_zero_decimal_currency(self):
        self.post("jpy-i", "invoice", "1000", "JY", currency="JPY")
        self.post("jpy-p", "payment", "1000", "JY", currency="JPY")
        batch = self.w.clearing.propose(
            "jpy", "a", "JPY", "2026-02-01", ["jpy-p"], ["jpy-i"]
        )
        self.assertEqual(batch["report"]["conservation"]["source_available_minor"], 1000)
        self.w.clearing.confirm("jpy", "treasurer")
        self.assertEqual(self.w.ledger.balance("jpy-i"), 0)
        self.assertTrue(self.w.controls.require_valid()["ok"])

    def test_dispatch_exposes_clearing_actions(self):
        self.post("i1", "invoice", "10", "R")
        self.post("p1", "payment", "10", "R")
        result = self.w.dispatch({
            "action": "clearing.propose",
            "arguments": {
                "batch_id": "d1", "party": "a", "currency": "CNY",
                "effective": "2026-02-01", "sources": ["p1"], "invoices": ["i1"],
            },
        })
        self.assertEqual(result["state"], "proposed")
        defaults = self.w.dispatch({"action": "clearing.weight_defaults", "arguments": {}})
        self.assertEqual(defaults["weights"], DEFAULT_WEIGHTS)
        with self.assertRaises(BookError):
            self.w.dispatch({"action": "clearing.unknown"})
