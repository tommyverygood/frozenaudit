"""Pre-declared gates, with a third verdict for "could not be evaluated".

A gate is a rule written down *before* any score is read: a quantity, a
threshold it must clear, and the minimum evidence needed for the comparison to
mean anything. Evaluating it afterwards returns one of three verdicts:

    pass           -- the declared criterion was met
    fail           -- the criterion was not met
    not_evaluable  -- the evidence needed to decide was not present

The third verdict is the point of this module. Reporting *fail* when a stratum
held too few molecules to estimate anything reads as evidence against the
model, when in fact nothing was learned either way. In the accompanying paper
two of three out-of-distribution gates came back ``not_evaluable`` because the
nearest stratum held 7 molecules; one passed; and on a separate strict blind
pool the binary-call gate came back ``fail``. All three verdicts are real
outcomes and all three are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

PASS, FAIL, NOT_EVALUABLE = "pass", "fail", "not_evaluable"


@dataclass
class Gate:
    """One pre-declared criterion.

    Parameters
    ----------
    name
        Short identifier used in reports.
    declared
        The criterion in words, exactly as written down before scoring. Kept
        verbatim so a reader can check that it was not adjusted afterwards.
    min_n
        Minimum number of evaluable rows. Below this the verdict is
        ``not_evaluable`` regardless of the estimate.
    rule
        Callable receiving the observation dict and returning True for pass.
    requires
        Keys the observation must carry and have non-None. A missing key gives
        ``not_evaluable``, never ``fail``.
    """

    name: str
    declared: str
    min_n: int
    rule: Callable[[dict], bool]
    requires: Sequence[str] = field(default_factory=tuple)

    def evaluate(self, observation: dict) -> dict:
        n = observation.get("n")
        missing = [k for k in self.requires if observation.get(k) is None]
        if n is None:
            return self._verdict(NOT_EVALUABLE, observation,
                                 "no sample size reported")
        if n < self.min_n:
            return self._verdict(NOT_EVALUABLE, observation,
                                 f"n = {n} below the declared minimum of {self.min_n}")
        if missing:
            return self._verdict(NOT_EVALUABLE, observation,
                                 f"missing required quantities: {sorted(missing)}")
        try:
            met = bool(self.rule(observation))
        except Exception as exc:                     # noqa: BLE001
            return self._verdict(NOT_EVALUABLE, observation,
                                 f"rule could not be applied: {type(exc).__name__}")
        return self._verdict(PASS if met else FAIL, observation,
                             "declared criterion met" if met
                             else "declared criterion not met")

    def _verdict(self, verdict: str, observation: dict, why: str) -> dict:
        return {"gate": self.name, "declared": self.declared, "verdict": verdict,
                "reason": why, "observation": dict(observation)}


def ci_lower_above(threshold: float = 0.0) -> Callable[[dict], bool]:
    """Gate rule: the 95% lower bound must exceed ``threshold``.

    This is the form used for the directional gates in the paper. It is
    deliberately one-sided: an interval that straddles the threshold does not
    pass, and it also does not become ``not_evaluable`` -- it is a genuine
    failure to demonstrate the effect.
    """
    return lambda obs: obs["ci95_low"] > threshold


def point_above(threshold: float) -> Callable[[dict], bool]:
    """Gate rule: the point estimate must exceed ``threshold``.

    Weaker than :func:`ci_lower_above` because it ignores precision. Use it
    only where the pre-declared rule was itself written on the point estimate.
    """
    return lambda obs: obs["estimate"] > threshold


def evaluate_all(gates: Sequence[Gate], observations: dict) -> list[dict]:
    """Evaluate every gate against a mapping of gate name to observation."""
    return [g.evaluate(observations.get(g.name, {})) for g in gates]


def summarise(results: Sequence[dict]) -> dict:
    """Counts per verdict, plus the overall reading.

    The overall reading is deliberately strict: any ``fail`` makes the set
    fail, and a set with no failures but some ``not_evaluable`` is reported as
    ``partial`` rather than as a pass. A protocol that reported "passed" while
    two of its three gates could not be evaluated would be overstating what it
    had shown.
    """
    counts = {PASS: 0, FAIL: 0, NOT_EVALUABLE: 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    if counts[FAIL]:
        overall = FAIL
    elif counts[NOT_EVALUABLE]:
        overall = "partial"
    elif counts[PASS]:
        overall = PASS
    else:
        overall = NOT_EVALUABLE
    return {"counts": counts, "overall": overall, "n_gates": len(results)}
