"""The core must run with no chemistry in sight.

These tests are the claim "``frozenaudit.core`` is endpoint-agnostic" written
as something that can fail. They use synthetic scores and labels with no
molecules, no RDKit and no model bundles, so a future change that sneaks an
ABCB1 assumption into the core breaks them.

They establish that the code is general. They do not establish that the
protocol's *conclusions* transfer to another endpoint -- no test can do that,
only a second worked instance can. See the note in ``frozenaudit/__init__``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frozenaudit.core import Audit, contrast, freeze, gates  # noqa: E402


# ----------------------------------------------------------------- freeze
def test_freeze_is_deterministic_across_key_order():
    """Two dicts differing only in insertion order freeze to the same hash."""
    a = {"threshold": 0.7, "metric": "balanced_accuracy", "n": 3}
    b = {"n": 3, "metric": "balanced_accuracy", "threshold": 0.7}
    assert freeze.freeze_hash(a) == freeze.freeze_hash(b)


def test_freeze_separates_close_floats():
    """A threshold nudged in the 7th decimal is a different protocol."""
    assert freeze.freeze_hash({"t": 0.70}) != freeze.freeze_hash({"t": 0.7000001})


def test_freeze_refuses_unknown_types():
    """An object with no canonical form raises rather than hashing its repr."""
    class Opaque:
        pass

    with pytest.raises(TypeError, match="no canonical form"):
        freeze.freeze_hash({"model": Opaque()})


def test_lock_verify_raises_on_drift():
    lock = freeze.FreezeLock.record("p", {"t": 0.5})
    assert lock.verify({"t": 0.5}) is True
    assert lock.short == lock.sha256[:16]
    with pytest.raises(freeze.FreezeMismatch):
        lock.verify({"t": 0.6})


# --------------------------------------------------------------- contrast
def _synthetic(n=800, seed=0):
    """Scores correlated with labels, plus a selector that prefers extremes."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    scores = np.clip(0.5 + 0.25 * (labels * 2 - 1) + rng.normal(0, 0.22, n), 0, 1)
    return scores, labels


def test_native_margin_arm_has_matched_n():
    """The baseline arm holds exactly as many rows as the selector accepted."""
    scores, labels = _synthetic()
    accepted = np.zeros(scores.size, dtype=bool)
    accepted[:150] = True
    out = contrast.coverage_matched_contrast(
        scores=scores, labels=labels, threshold=0.5, selector_accepted=accepted)
    assert out["arms"]["selected"]["n"] == 150
    assert out["arms"]["native_margin_matched"]["n"] == 150
    assert out["arms"]["selected"]["coverage"] == pytest.approx(150 / 800)


def test_margin_selection_is_stable_under_ties():
    """All-tied margins select the first n rows, not an arbitrary n."""
    scores = np.full(10, 0.8)
    sel = contrast.select_by_native_margin(scores, threshold=0.5, n=4)
    assert sel.tolist() == [True] * 4 + [False] * 6


def test_nan_score_is_never_confident():
    """A missing score sorts last: absence of a score is not certainty."""
    scores = np.array([0.9, np.nan, 0.85, 0.55])
    sel = contrast.select_by_native_margin(scores, threshold=0.5, n=2)
    assert sel.tolist() == [True, False, True, False]


def test_small_arm_is_not_evaluable_and_reports_no_number():
    """Below min_n an arm carries None for every metric, never an estimate."""
    scores, labels = _synthetic()
    accepted = np.zeros(scores.size, dtype=bool)
    accepted[:9] = True
    out = contrast.coverage_matched_contrast(
        scores=scores, labels=labels, threshold=0.5, selector_accepted=accepted,
        min_n=30)
    arm = out["arms"]["selected"]
    assert arm["evaluable"] is False
    assert all(v is None for v in arm["metrics"].values())
    assert "below the declared minimum" in arm["why"]
    assert out["deltas"]["balanced_accuracy"]["vs_native_margin"] is None


def test_single_class_arm_gives_none_not_half():
    """A one-class arm reports None balanced accuracy rather than 0.5."""
    y = np.ones(50, dtype=int)
    assert contrast.balanced_accuracy(y, np.ones(50, dtype=int)) is None


def test_reading_names_the_winner_including_when_it_is_the_model():
    """When the model's own margin wins, the reading says so in words."""
    scores, labels = _synthetic()
    # a deliberately poor selector: rows nearest the threshold
    m = np.abs(scores - 0.5)
    accepted = np.zeros(scores.size, dtype=bool)
    accepted[np.argsort(m)[:200]] = True
    out = contrast.coverage_matched_contrast(
        scores=scores, labels=labels, threshold=0.5, selector_accepted=accepted,
        selector_name="bad_selector")
    assert "the model's own margin beats bad_selector" in out["reading"]
    assert out["deltas"]["balanced_accuracy"]["vs_native_margin"] < 0


def test_missing_labels_are_dropped_and_counted():
    scores, labels = _synthetic(n=100)
    lab = labels.astype(float)
    lab[:7] = np.nan
    accepted = np.zeros(100, dtype=bool)
    accepted[:60] = True
    out = contrast.coverage_matched_contrast(
        scores=scores, labels=lab, threshold=0.5, selector_accepted=accepted,
        min_n=10)
    assert out["dropped"] == 7 and out["n_total"] == 93


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        contrast.coverage_matched_contrast(
            scores=[0.1, 0.2], labels=[0, 1, 0], threshold=0.5,
            selector_accepted=[True, False])


# --------------------------------------------------------------- protocol
def _audit_with_gate():
    return Audit("synthetic_endpoint", gates=[
        gates.Gate("direction", "95% lower bound above zero", min_n=30,
                   rule=gates.ci_lower_above(0.0), requires=("ci95_low",))])


def test_audit_runs_with_no_pool_and_no_chemistry():
    """Steps 1, 3, 4, 5 complete without a reference pool or RDKit."""
    scores, labels = _synthetic()
    accepted = np.zeros(scores.size, dtype=bool)
    accepted[:200] = True
    audit = _audit_with_gate()
    audit.freeze({"threshold": 0.5, "metric": "balanced_accuracy"})
    audit.contrast(scores=scores, labels=labels, threshold=0.5,
                   selector_accepted=accepted)
    audit.run_gates({"direction": {"n": 200, "ci95_low": 0.12}})
    rep = audit.report()
    assert rep["declared_before_scoring"] is True
    assert rep["overall"] == "pass"
    assert rep["steps_missing"] == []
    assert rep["freeze"]["short"] == audit.lock.short


def test_scoring_before_freezing_is_downgraded_to_exploratory():
    """A gate that passes in an unfrozen run is reported as exploratory."""
    audit = _audit_with_gate()
    audit.run_gates({"direction": {"n": 200, "ci95_low": 0.12}})
    audit.freeze({"threshold": 0.5})
    rep = audit.report()
    assert rep["declared_before_scoring"] is False
    assert rep["overall"] == "exploratory"
    assert any("not frozen before scoring" in c for c in rep["caveats"])


def test_never_frozen_is_also_exploratory():
    audit = _audit_with_gate()
    audit.run_gates({"direction": {"n": 200, "ci95_low": 0.12}})
    assert audit.report()["overall"] == "exploratory"


def test_missing_step_is_incomplete_not_pass():
    audit = _audit_with_gate()
    audit.freeze({"threshold": 0.5})
    audit.run_gates({"direction": {"n": 200, "ci95_low": 0.12}})
    rep = audit.report()          # contrast never run
    assert rep["overall"] == "incomplete"
    assert "contrast" in rep["steps_missing"]


def test_not_evaluable_gate_never_reads_as_pass():
    audit = _audit_with_gate()
    audit.freeze({"threshold": 0.5})
    scores, labels = _synthetic()
    accepted = np.zeros(scores.size, dtype=bool)
    accepted[:200] = True
    audit.contrast(scores=scores, labels=labels, threshold=0.5,
                   selector_accepted=accepted)
    audit.run_gates({"direction": {"n": 7, "ci95_low": 0.9}})
    rep = audit.report()
    assert rep["overall"] != "pass"
    assert any("not evaluable" in c for c in rep["caveats"])


def test_freeze_covers_the_gate_declarations():
    """Editing a gate's declared wording after freezing is detectable."""
    a = _audit_with_gate()
    lock = a.freeze({"threshold": 0.5})
    b = Audit("synthetic_endpoint", gates=[
        gates.Gate("direction", "95% lower bound above zero, revised",
                   min_n=30, rule=gates.ci_lower_above(0.0),
                   requires=("ci95_low",))])
    with pytest.raises(freeze.FreezeMismatch):
        lock.verify({"threshold": 0.5,
                     "gates": [{"name": g.name, "declared": g.declared,
                                "min_n": g.min_n, "requires": list(g.requires)}
                               for g in b.gates]})


def test_stratify_without_pool_raises_clearly():
    audit = _audit_with_gate()
    with pytest.raises(ValueError, match="needs a reference pool"):
        audit.stratify(["CCO"])
