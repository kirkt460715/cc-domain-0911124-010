"""批次级核销建议与确认。

建议阶段只读账本：把同一客户、同一币种、同一业务截止日下的多笔回款与
开放发票做成不可变快照，交给纯求解器全局求解。确认阶段在同一事务内逐项
重验可用余额、争议与期间状态，任一处过时整批拒绝；重复确认不重复入账，
失败保留原建议与原因。
"""

from ..primitives import (
    identifier,
    day,
    display,
    digest,
    canonical,
    decode,
    error,
    integer,
    obj,
    sequence,
)
from .clearing_solver import DEFAULT_WEIGHTS, solve


DDL = """
CREATE TABLE IF NOT EXISTS clearing_weights(id TEXT PRIMARY KEY,version INTEGER NOT NULL,
 payload TEXT NOT NULL,digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS clearing_batches(id TEXT PRIMARY KEY,revision INTEGER NOT NULL,
 state TEXT NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL,result TEXT);
CREATE TABLE IF NOT EXISTS clearing_revisions(id TEXT PRIMARY KEY,batch TEXT NOT NULL,
 revision INTEGER NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS clearing_locks(batch TEXT NOT NULL,source TEXT NOT NULL,
 invoice TEXT NOT NULL,amount INTEGER NOT NULL,locked_by TEXT NOT NULL,
 note TEXT NOT NULL,PRIMARY KEY(batch,source,invoice));
CREATE TABLE IF NOT EXISTS clearing_failures(id TEXT PRIMARY KEY,batch TEXT NOT NULL,
 revision INTEGER NOT NULL,reason TEXT NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL);
"""


class BatchClearing:
    def __init__(self, database, ledger):
        self.db = database
        self.ledger = ledger
        self.db.connection.executescript(DDL)

    # ---------- 目标与证据权重版本 ----------

    def weight_save(self, weight_id, weights, expected_version=0):
        identifier(weight_id)
        integer(expected_version, 0)
        normalized = {}
        obj(weights, set(DEFAULT_WEIGHTS), set(DEFAULT_WEIGHTS))
        for key in DEFAULT_WEIGHTS:
            normalized[key] = integer(weights[key], 0, 1000)
        payload = {"id": weight_id, "weights": normalized}
        with self.db.transaction() as con:
            old = self.db.one(
                "SELECT * FROM clearing_weights WHERE id=?", (weight_id,)
            )
            version = old["version"] if old else 0
            if expected_version != version:
                error("conflict", "weight profile version changed")
            version += 1
            con.execute(
                "INSERT INTO clearing_weights VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET version=excluded.version,"
                "payload=excluded.payload,digest=excluded.digest",
                (weight_id, version, canonical(payload), digest(payload)),
            )
            self.db.audit(
                "clearing_weight_saved",
                weight_id,
                {"version": version, "weights": normalized},
            )
        return self.weight_get(weight_id)

    def weight_get(self, weight_id):
        row = self.db.one(
            "SELECT * FROM clearing_weights WHERE id=?", (weight_id,)
        )
        if row is None:
            error("missing", "clearing weight profile not found")
        payload = decode(row["payload"])
        if digest(payload) != row["digest"]:
            error("integrity", "clearing weight fingerprint differs")
        return {**payload, "version": row["version"]}

    def weight_defaults(self):
        return {"id": "default", "version": 1, "weights": dict(DEFAULT_WEIGHTS)}

    def _resolve_weights(self, weight_id):
        if weight_id is None or weight_id == "default":
            return dict(DEFAULT_WEIGHTS), self.weight_defaults()
        return self.weight_get(weight_id)["weights"], self.weight_get(weight_id)

    # ---------- 快照（只读账本） ----------

    def _snapshot(self, party, currency, effective, sources, invoices, weight_id):
        self.ledger.party(party)
        source_rows, invoice_rows = [], []
        seen_sources, seen_invoices = set(), set()
        for source_id in sequence(sources, 1, 100):
            identifier(source_id)
            document = self.ledger.document(source_id)
            if (
                document["party"],
                document["currency"],
            ) != (party, currency) or document["kind"] not in {"payment", "credit"}:
                error("input", "batch source party, currency or kind differs")
            if source_id in seen_sources:
                error("duplicate", "duplicate source in batch")
            seen_sources.add(source_id)
            available = self.ledger.available_from(source_id, effective)
            if available <= 0:
                error("input", "batch source has no available amount", source=source_id)
            source_rows.append(
                {
                    "id": source_id,
                    "kind": document["kind"],
                    "reference": document["reference"],
                    "available_minor": available,
                    "balance_baseline": self.ledger.balance(source_id),
                }
            )
        for invoice_id in sequence(invoices, 1, 100):
            identifier(invoice_id)
            document = self.ledger.document(invoice_id)
            if (
                document["party"],
                document["currency"],
            ) != (party, currency) or document["kind"] != "invoice":
                error("input", "batch invoice party, currency or kind differs")
            if invoice_id in seen_invoices:
                error("duplicate", "duplicate invoice in batch")
            seen_invoices.add(invoice_id)
            available = self.ledger.available_from(invoice_id, effective)
            if available <= 0:
                error(
                    "input",
                    "batch invoice has no open balance",
                    invoice=invoice_id,
                )
            invoice_rows.append(
                {
                    "id": invoice_id,
                    "reference": document["reference"],
                    "due": document["due"],
                    "open_minor": available,
                    "balance_baseline": self.ledger.balance(invoice_id),
                }
            )
        weights, weight_profile = self._resolve_weights(weight_id)
        dispute_digest = digest(
            self.db.rows(
                "SELECT invoice,amount,opened,closed,reason,revision FROM disputes "
                "ORDER BY invoice,opened",
            )
        )
        snapshot = {
            "party": party,
            "currency": currency,
            "effective": effective,
            "sources": source_rows,
            "invoices": invoice_rows,
            "weight_profile": weight_profile,
            "dispute_digest": dispute_digest,
        }
        return snapshot, weights

    def _locks_for(self, batch_id):
        return [
            {
                "source": row["source"],
                "invoice": row["invoice"],
                "amount_minor": row["amount"],
            }
            for row in self.db.rows(
                "SELECT * FROM clearing_locks WHERE batch=? ORDER BY source,invoice",
                (batch_id,),
            )
        ]

    # ---------- 建议 ----------

    def propose(
        self,
        batch_id,
        party,
        currency,
        effective,
        sources,
        invoices,
        weight_id="default",
        dropped=None,
        maximum_alternatives=5,
    ):
        identifier(batch_id)
        effective = day(effective)
        integer(maximum_alternatives, 1, 20)
        dropped = sequence(dropped or [], 0, 200)
        normalized_dropped = []
        for row in dropped:
            obj(row, {"source", "invoice"}, {"source", "invoice"})
            normalized_dropped.append(
                {"source": identifier(row["source"]), "invoice": identifier(row["invoice"])}
            )
        snapshot, weights = self._snapshot(
            party, currency, effective, sources, invoices, weight_id
        )
        report = solve(
            snapshot["sources"],
            snapshot["invoices"],
            effective,
            weights,
            locks=self._locks_for(batch_id),
            dropped=normalized_dropped,
            maximum_alternatives=maximum_alternatives,
        )
        payload = {
            "id": batch_id,
            "snapshot": snapshot,
            "dropped": normalized_dropped,
            "report": report,
            "request_digest": digest(
                {
                    "party": party,
                    "currency": currency,
                    "effective": effective,
                    "sources": sorted(sources),
                    "invoices": sorted(invoices),
                    "weight_id": snapshot["weight_profile"]["id"],
                    "weight_version": snapshot["weight_profile"]["version"],
                    "dropped": normalized_dropped,
                    "locks": self._locks_for(batch_id),
                }
            ),
        }
        with self.db.transaction() as con:
            old = self.db.one(
                "SELECT * FROM clearing_batches WHERE id=?", (batch_id,)
            )
            if old:
                if old["state"] != "proposed":
                    error("state", "confirmed or failed batch cannot be re-proposed")
                previous = decode(old["payload"])
                if previous["request_digest"] != payload["request_digest"]:
                    error("conflict", "clearing batch identity differs")
                return self.get(batch_id)
            revision = 1
            con.execute(
                "INSERT INTO clearing_batches VALUES(?,?,?,?,?,NULL)",
                (
                    batch_id,
                    revision,
                    "proposed",
                    canonical(payload),
                    digest(payload),
                ),
            )
            con.execute(
                "INSERT INTO clearing_revisions VALUES(?,?,?,?,?)",
                (batch_id + ":1", batch_id, revision, canonical(payload), digest(payload)),
            )
            self.db.audit(
                "clearing_proposed",
                batch_id,
                {
                    "revision": revision,
                    "objective": report["objective"],
                    "ambiguous": report["ambiguous"],
                },
            )
        return self.get(batch_id)

    def get(self, batch_id):
        row = self.db.one(
            "SELECT * FROM clearing_batches WHERE id=?", (batch_id,)
        )
        if row is None:
            error("missing", "clearing batch not found")
        payload = decode(row["payload"])
        if digest(payload) != row["digest"]:
            error("integrity", "clearing batch fingerprint differs")
        result = decode(row["result"]) if row["result"] else None
        return {
            **payload,
            "revision": row["revision"],
            "state": row["state"],
            "result": result,
            "digest": row["digest"],
        }

    def revisions(self, batch_id):
        self.get(batch_id)
        return [
            {
                "id": row["id"],
                "revision": row["revision"],
                "digest": row["digest"],
            }
            for row in self.db.rows(
                "SELECT * FROM clearing_revisions WHERE batch=? ORDER BY revision",
                (batch_id,),
            )
        ]

    def failures(self, batch_id):
        self.get(batch_id)
        rows = self.db.rows(
            "SELECT id,revision,reason,payload FROM clearing_failures "
            "WHERE batch=? ORDER BY id",
            (batch_id,),
        )
        return [
            {**row, "payload": decode(row["payload"])} for row in rows
        ]

    def recompute(self, batch_id, dropped=None, maximum_alternatives=5):
        """锁定已人工确认的部分后，对其余部分重新求解，生成新修订版本。"""
        identifier(batch_id)
        integer(maximum_alternatives, 1, 20)
        dropped = sequence(dropped or [], 0, 200)
        normalized_dropped = []
        for row in dropped:
            obj(row, {"source", "invoice"}, {"source", "invoice"})
            normalized_dropped.append(
                {"source": identifier(row["source"]), "invoice": identifier(row["invoice"])}
            )
        with self.db.transaction() as con:
            current = self.get(batch_id)
            if current["state"] != "proposed":
                error("state", "only proposed batch can be recomputed")
            snapshot = current["snapshot"]
            weights = snapshot["weight_profile"]["weights"]
            report = solve(
                snapshot["sources"],
                snapshot["invoices"],
                snapshot["effective"],
                weights,
                locks=self._locks_for(batch_id),
                dropped=normalized_dropped,
                maximum_alternatives=maximum_alternatives,
            )
            payload = {
                "id": batch_id,
                "snapshot": snapshot,
                "dropped": normalized_dropped,
                "report": report,
                "request_digest": current["request_digest"],
            }
            revision = current["revision"] + 1
            con.execute(
                "UPDATE clearing_batches SET revision=?,payload=?,digest=? WHERE id=?",
                (revision, canonical(payload), digest(payload), batch_id),
            )
            con.execute(
                "INSERT INTO clearing_revisions VALUES(?,?,?,?,?)",
                (
                    batch_id + ":" + str(revision),
                    batch_id,
                    revision,
                    canonical(payload),
                    digest(payload),
                ),
            )
            self.db.audit(
                "clearing_recomputed",
                batch_id,
                {"revision": revision, "objective": report["objective"]},
            )
        return self.get(batch_id)

    # ---------- 人工锁定 ----------

    def lock(self, batch_id, source, invoice, amount, operator, note):
        identifier(batch_id)
        identifier(operator)
        if not isinstance(note, str) or not note.strip() or len(note) > 1000:
            error("input", "lock requires bounded explanation")
        batch = self.get(batch_id)
        if batch["state"] != "proposed":
            error("state", "only proposed batch can be locked")
        source_ids = {row["id"] for row in batch["snapshot"]["sources"]}
        invoice_ids = {row["id"] for row in batch["snapshot"]["invoices"]}
        if source not in source_ids or invoice not in invoice_ids:
            error("input", "lock references document outside batch")
        code = batch["snapshot"]["currency"]
        amount_minor = self._minor(amount, code)
        with self.db.transaction() as con:
            existing = self.db.one(
                "SELECT * FROM clearing_locks WHERE batch=? AND source=? AND invoice=?",
                (batch_id, source, invoice),
            )
            if existing:
                if (
                    existing["amount"],
                    existing["locked_by"],
                    existing["note"],
                ) != (amount_minor, operator, note.strip()):
                    error("conflict", "lock identity differs")
                return {"existing": True}
            con.execute(
                "INSERT INTO clearing_locks VALUES(?,?,?,?,?,?)",
                (batch_id, source, invoice, amount_minor, operator, note.strip()),
            )
            self.db.audit(
                "clearing_locked",
                batch_id,
                {
                    "source": source,
                    "invoice": invoice,
                    "amount_minor": amount_minor,
                    "operator": operator,
                },
            )
        return {"existing": False}

    def unlock(self, batch_id, source, invoice, reason):
        if not isinstance(reason, str) or not reason.strip():
            error("input", "unlock requires reason")
        self.get(batch_id)
        with self.db.transaction() as con:
            existing = self.db.one(
                "SELECT * FROM clearing_locks WHERE batch=? AND source=? AND invoice=?",
                (batch_id, source, invoice),
            )
            if existing is None:
                error("missing", "lock not found")
            con.execute(
                "DELETE FROM clearing_locks WHERE batch=? AND source=? AND invoice=?",
                (batch_id, source, invoice),
            )
            self.db.audit(
                "clearing_unlocked",
                batch_id,
                {"source": source, "invoice": invoice, "reason": reason},
            )
        return {"unlocked": existing["amount"]}

    def locks(self, batch_id):
        self.get(batch_id)
        return self.db.rows(
            "SELECT * FROM clearing_locks WHERE batch=? ORDER BY source,invoice",
            (batch_id,),
        )

    # ---------- 去掉候选边后的影响分析 ----------

    def edge_impact(self, batch_id, source, invoice):
        identifier(source)
        identifier(invoice)
        batch = self.get(batch_id)
        if batch["state"] != "proposed":
            error("state", "impact analysis requires proposed batch")
        key = source + "->" + invoice
        current_edges = {row["edge"] for row in batch["report"]["edges"]}
        locked_keys = {row["edge"] for row in batch["report"]["locked"]}
        if key not in current_edges and key not in locked_keys:
            error("missing", "edge not present in batch")
        snapshot = batch["snapshot"]
        alternative = solve(
            snapshot["sources"],
            snapshot["invoices"],
            snapshot["effective"],
            snapshot["weight_profile"]["weights"],
            locks=[
                row
                for row in self._locks_for(batch_id)
                if not (row["source"] == source and row["invoice"] == invoice)
            ] if key in locked_keys else self._locks_for(batch_id),
            dropped=batch["dropped"] + [{"source": source, "invoice": invoice}],
        )
        before = batch["report"]["objective"]
        after = alternative["objective"]
        return {
            "batch": batch_id,
            "removed_edge": key,
            "objective_before": before,
            "objective_after": after,
            "deltas": {
                "evidence_score_change": after["evidence_score"]
                - before["evidence_score"],
                "unapplied_change_minor": after["unapplied_minor"]
                - before["unapplied_minor"],
                "splits_change": after["splits"] - before["splits"],
            },
            "still_feasible": alternative["conservation"]["balanced"],
            "weak_evidence": key in batch["report"]["weak_evidence_edges"],
            "alternative_digest": alternative["solution_digest"],
        }

    # ---------- 确认：同一事务逐项重验 ----------

    @staticmethod
    def _minor(amount, code):
        from ..primitives import money

        return money(amount, code)

    def confirm(self, batch_id, operator):
        identifier(operator)
        identifier(batch_id)
        batch = self.get(batch_id)
        if batch["state"] == "confirmed":
            # 重复确认不重复入账。
            return batch
        if batch["state"] != "proposed":
            error("state", "only proposed batch can be confirmed")
        snapshot = batch["snapshot"]
        effective = snapshot["effective"]
        try:
            with self.db.transaction() as con:
                self.ledger.check_period(effective)
                if digest(
                    self.db.rows(
                        "SELECT invoice,amount,opened,closed,reason,revision FROM disputes "
                        "ORDER BY invoice,opened"
                    )
                ) != snapshot["dispute_digest"]:
                    error("conflict", "dispute register changed since proposal")
                for source in snapshot["sources"]:
                    if self.ledger.balance(source["id"]) != source["balance_baseline"]:
                        error(
                            "conflict",
                            "source balance changed since proposal",
                            source=source["id"],
                        )
                for invoice in snapshot["invoices"]:
                    if self.ledger.balance(invoice["id"]) != invoice["balance_baseline"]:
                        error(
                            "conflict",
                            "invoice balance changed since proposal",
                            invoice=invoice["id"],
                        )
                # 逐项重验可用余额：在本事务内按顺序累计扣减，确保每条
                # 分配（含锁定与建议部分）的剩余需求不超过剩余可用额。
                remaining_source = {
                    source["id"]: self.ledger.available_from(
                        source["id"], effective
                    )
                    for source in snapshot["sources"]
                }
                remaining_invoice = {
                    invoice["id"]: self.ledger.available_from(
                        invoice["id"], effective
                    )
                    for invoice in snapshot["invoices"]
                }
                for item in batch["report"]["allocations"]:
                    needed = item["amount_minor"]
                    if (
                        needed > remaining_source[item["source"]]
                        or needed > remaining_invoice[item["invoice"]]
                    ):
                        error(
                            "conflict",
                            "allocation no longer covered by available balance",
                            edge=item["edge"],
                        )
                    remaining_source[item["source"]] -= needed
                    remaining_invoice[item["invoice"]] -= needed
                code = snapshot["currency"]
                posted = []
                for index, item in enumerate(batch["report"]["allocations"]):
                    allocation_id = "clear:" + batch_id + ":" + str(index)
                    self.ledger.allocate(
                        allocation_id,
                        item["source"],
                        item["invoice"],
                        display(item["amount_minor"], code),
                        effective,
                        "clearing:" + batch_id,
                    )
                    posted.append(allocation_id)
                # 事务内守恒核对：按币种最小单位。
                if not batch["report"]["conservation"]["balanced"]:
                    error("integrity", "proposed batch does not conserve money")
                result = {
                    "operator": operator,
                    "allocations": posted,
                    "revision": batch["revision"],
                }
                con.execute(
                    "UPDATE clearing_batches SET state='confirmed',result=? WHERE id=?",
                    (canonical(result), batch_id),
                )
                self.db.audit("clearing_confirmed", batch_id, result)
        except Exception as exc:
            # 一处过时整批拒绝：业务事务已整体回滚，原建议保持 proposed；
            # 另开事务仅追加失败原因留痕。
            from ..primitives import BookError

            reason = exc.code if isinstance(exc, BookError) else "unexpected"
            details = (
                exc.details if isinstance(exc, BookError) else {"message": str(exc)}
            )
            self._record_failure(batch, reason, details)
            raise
        return self.get(batch_id)

    def _record_failure(self, batch, reason, details):
        payload = {
            "batch": batch["id"],
            "revision": batch["revision"],
            "reason": reason,
            "details": details,
        }
        with self.db.transaction():
            sequence_row = self.db.one(
                "SELECT COUNT(*) AS n FROM clearing_failures WHERE batch=?",
                (batch["id"],),
            )
            failure_id = (
                "fail:" + batch["id"] + ":" + str(batch["revision"]) + ":"
                + str(sequence_row["n"] + 1)
            )
            self.db.connection.execute(
                "INSERT INTO clearing_failures VALUES(?,?,?,?,?,?)",
                (
                    failure_id,
                    batch["id"],
                    batch["revision"],
                    reason,
                    canonical(payload),
                    digest(payload),
                ),
            )
            self.db.audit("clearing_confirm_failed", batch["id"], payload)
