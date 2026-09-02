"""The five steps, in order, with a record of whether they were actually taken.

    1. freeze     -- write the rules down and hash them          (:mod:`.freeze`)
    2. stratify   -- structure-only distance to a reference pool (:mod:`.domain`)
    3. contrast   -- coverage-matched, against the model's own confidence
                                                                (:mod:`.contrast`)
    4. gates      -- pre-declared criteria, three verdicts        (:mod:`.gates`)
    5. report     -- emit all of it, including what was not evaluable

:class:`Audit` is a thin orchestrator over those four modules. It adds one
thing they cannot provide separately: **an order of events**.

Why order is the point
----------------------
Every element of an audit protocol can be honest in isolation and the whole
still be circular, if the criteria were written after the numbers were seen.
That is not detectable by inspecting the criteria -- a threshold chosen to be
just below the observed value looks exactly like a threshold chosen in
advance. It is detectable by recording when each step happened.

So :class:`Audit` timestamps and sequences its steps, and
:meth:`Audit.report` states plainly whether the freeze preceded the scoring.
When it did not, the report carries ``declared_before_scoring: False`` and the
overall reading is downgraded to ``exploratory`` no matter how the gates came
out. A gate that passes in an exploratory run is a hypothesis, not a result,
and the report says so rather than leaving the reader to work it out.

This does not make circularity impossible -- someone can always run the
protocol twice and keep the second lock. It makes the honest path the
default one, and it makes the dishonest path require an explicit act.

Nothing in this module is chemistry-specific except :meth:`Audit.stratify`,
which needs a chemical reference pool; skip that step and the rest still runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import contrast as _contrast
from . import gates as _gates
from .freeze import FreezeLock

__all__ = ["Audit", "AuditStep"]


@dataclass
class AuditStep:
    """One step actually taken, with the wall-clock order it happened in."""

    name: str
    at: float
    detail: dict = field(default_factory=dict)


class Audit:
    """An audit in progress.

    Parameters
    ----------
    name
        Identifier carried into the report.
    pool
        Optional reference pool for :meth:`stratify`, in the form
        :func:`frozenaudit.core.domain.build_pool` returns. Omit for a
        non-chemical audit or when you only want steps 1, 3, 4.
    gates
        The pre-declared criteria. Pass them at construction time -- that is
        the point -- rather than assembling them after seeing a contrast.
    """

    def __init__(self, name: str, *, pool: Sequence[dict] | None = None,
                 gates: Sequence[_gates.Gate] = ()) -> None:
        self.name = name
        self.pool = list(pool) if pool is not None else None
        self.gates = list(gates)
        self.lock: FreezeLock | None = None
        self.steps: list[AuditStep] = []
        self._thresholds: dict | None = None
        self._strata: list[dict] | None = None
        self._contrast: dict | None = None
        self._gate_results: list[dict] | None = None

    # ---------------------------------------------------------------- step 1
    def freeze(self, protocol: dict, note: str = "") -> FreezeLock:
        """Hash the protocol description and record that it happened.

        ``protocol`` should contain everything that could change a number:
        thresholds, the gate declarations, the reference-pool identity, the
        metric names. It does not need to contain the data.
        """
        payload = dict(protocol)
        payload.setdefault("gates", [{"name": g.name, "declared": g.declared,
                                      "min_n": g.min_n,
                                      "requires": list(g.requires)}
                                     for g in self.gates])
        if self.pool is not None:
            payload.setdefault("pool_size", len(self.pool))
        self.lock = FreezeLock.record(self.name, payload, note=note)
        self.steps.append(AuditStep("freeze", time.time(),
                                    {"sha256": self.lock.sha256,
                                     "short": self.lock.short}))
        return self.lock

    # ---------------------------------------------------------------- step 2
    def stratify(self, query_smiles: Sequence[str], k: int = 5) -> list[dict]:
        """Assign structure-only strata to ``query_smiles`` against the pool.

        Cut points come from the pool's own leave-one-connectivity-out D5
        distribution, so they are a property of the reference set rather than
        of the queries -- a query set cannot move its own goalposts.
        """
        if self.pool is None:
            raise ValueError(
                "stratify needs a reference pool; construct the Audit with "
                "pool=... or skip this step"
            )
        from . import domain  # local import: rdkit is only needed for this step

        loo = domain.leave_one_connectivity_out_d5(self.pool, k=k)
        self._thresholds = domain.d5_thresholds(loo)
        self._strata = domain.score_queries(query_smiles, self.pool,
                                            self._thresholds, k=k)
        counts: dict[str, int] = {}
        for row in self._strata:
            key = row["stratum"] if row["stratum"] is not None else "unparseable"
            counts[key] = counts.get(key, 0) + 1
        self.steps.append(AuditStep("stratify", time.time(),
                                    {"n_queries": len(self._strata),
                                     "thresholds": dict(self._thresholds),
                                     "stratum_counts": counts}))
        return self._strata

    # ---------------------------------------------------------------- step 3
    def contrast(self, **kwargs: Any) -> dict:
        """Run the four-arm coverage-matched contrast.

        Arguments are passed through to
        :func:`frozenaudit.core.contrast.coverage_matched_contrast`.
        """
        self._contrast = _contrast.coverage_matched_contrast(**kwargs)
        self.steps.append(AuditStep("contrast", time.time(),
                                    {"n_total": self._contrast["n_total"],
                                     "coverage": self._contrast["coverage"],
                                     "reading": self._contrast["reading"]}))
        return self._contrast

    # ---------------------------------------------------------------- step 4
    def run_gates(self, observations: dict) -> list[dict]:
        """Evaluate every declared gate against ``observations``.

        ``observations`` maps gate name to a dict carrying at least ``n`` and
        whatever keys that gate declared in ``requires``.
        """
        if not self.gates:
            raise ValueError("no gates were declared at construction time")
        self._gate_results = _gates.evaluate_all(self.gates, observations)
        summary = _gates.summarise(self._gate_results)
        self.steps.append(AuditStep("gates", time.time(), summary))
        return self._gate_results

    # ---------------------------------------------------------------- step 5
    @property
    def declared_before_scoring(self) -> bool:
        """Did :meth:`freeze` run before any of :meth:`contrast` / :meth:`run_gates`?

        False also when no freeze happened at all: an unfrozen protocol is not
        a pre-declared one.
        """
        order = [s.name for s in self.steps]
        if "freeze" not in order:
            return False
        first_freeze = order.index("freeze")
        scored = [i for i, n in enumerate(order) if n in ("contrast", "gates")]
        return all(i > first_freeze for i in scored)

    def report(self) -> dict:
        """Everything the audit established, plus everything it could not.

        The ``overall`` field is deliberately conservative:

        ``exploratory``
            the protocol was not frozen before scoring -- gate verdicts are
            hypotheses regardless of how they came out
        ``incomplete``
            frozen, but a required step was never run
        otherwise the gate summary's own reading (``pass`` / ``fail`` /
        ``partial``), which itself never returns ``pass`` when any gate came
        back ``not_evaluable``
        """
        gate_summary = (_gates.summarise(self._gate_results)
                        if self._gate_results is not None else None)
        run = [s.name for s in self.steps]
        missing = [s for s in ("freeze", "contrast", "gates") if s not in run]

        if not self.declared_before_scoring:
            overall = "exploratory"
        elif missing:
            overall = "incomplete"
        else:
            overall = gate_summary["overall"]

        caveats = []
        if not self.declared_before_scoring:
            caveats.append(
                "the protocol was not frozen before scoring; every verdict "
                "below is exploratory and must not be reported as a "
                "pre-declared result"
            )
        if missing:
            caveats.append(f"steps never run: {', '.join(missing)}")
        if self._contrast is not None:
            for key, arm in self._contrast["arms"].items():
                if not arm["evaluable"]:
                    caveats.append(f"contrast arm {arm['name']!r} not evaluable: "
                                   f"{arm['why']}")
        if self._gate_results is not None:
            for g in self._gate_results:
                if g["verdict"] == _gates.NOT_EVALUABLE:
                    caveats.append(f"gate {g['gate']!r} not evaluable: "
                                   f"{g['reason']}")

        return {
            "audit": self.name,
            "overall": overall,
            "declared_before_scoring": self.declared_before_scoring,
            "freeze": self.lock.as_dict() if self.lock else None,
            "steps_run": run,
            "steps_missing": missing,
            "strata": self.steps[run.index("stratify")].detail
                      if "stratify" in run else None,
            "contrast": self._contrast,
            "gates": {"results": self._gate_results, "summary": gate_summary}
                     if self._gate_results is not None else None,
            "caveats": caveats,
        }
