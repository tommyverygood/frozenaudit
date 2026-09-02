"""The correctness gate: this package must reproduce the frozen protocol.

These tests are not unit tests of convenience. If ``test_d5_matches_frozen``
fails, :mod:`frozenaudit.domain` has drifted from the protocol that produced the
published numbers, and the package -- not the frozen file -- is wrong.

The frozen files live outside the repository (they are the paper's deposit), so
each test skips when its input is absent. Point ``FROZENAUDIT_FROZEN_DIR`` at a
directory holding them to run the reconciliation.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from frozenaudit.core import domain, gates
from frozenaudit.instances.abcb1 import load_pool

FROZEN = Path(os.environ.get("FROZENAUDIT_FROZEN_DIR", ""))
MODEL = Path(__file__).resolve().parent.parent / "model"


def _pool():
    """The pool exactly as a user would obtain it.

    This deliberately calls the shipped loader rather than rebuilding the pool
    here. A test that constructs its own copy can pass while the function
    users actually call has drifted -- which is the failure mode this whole
    test module exists to prevent.
    """
    return load_pool()


def test_pool_dedup_size():
    """443 + 63 members collapse to 477 once stereoisomers are merged."""
    assert len(_pool()) == 477


def test_thresholds_match_frozen():
    """Q50/Q80/Q95 must equal the frozen leave-one-connectivity-out quantiles."""
    import pandas as pd

    path = FROZEN / "abcb1_ood_reference_loo_distances_v1.csv"
    if not path.exists():
        pytest.skip(f"frozen leave-one-out file not available at {path}")
    frozen = pd.read_csv(path)
    frozen = frozen[frozen.reference_pool == "joint_model_connectivity_unique"]
    mine = domain.d5_thresholds(domain.leave_one_connectivity_out_d5(_pool()))
    for q, p in (("q50", 0.50), ("q80", 0.80), ("q95", 0.95)):
        assert abs(mine[q] - float(np.quantile(frozen.loo_d5.values, p))) < 1e-12


def test_d5_matches_frozen():
    """Per-molecule D5 and stratum must match the frozen assignment file."""
    import pandas as pd

    path = FROZEN / "abcb1_ood_structural_assignments_v1.csv"
    if not path.exists():
        pytest.skip(f"frozen assignment file not available at {path}")
    frozen = pd.read_csv(path).head(400)
    pool = _pool()
    th = domain.d5_thresholds(domain.leave_one_connectivity_out_d5(pool))
    got = domain.score_queries(frozen.standardized_isomeric_smiles.tolist(), pool, th)
    d5 = np.array([r["d5"] for r in got], dtype=float)
    assert np.nanmax(np.abs(d5 - frozen.joint_model_d5.values)) < 1e-9
    assert all(g["stratum"] == f for g, f in
               zip(got, frozen.reference_adaptive_stratum))


def test_contrast_reproduces_published_table():
    """The general contrast must reproduce the paper's Table 4, to the last digit.

    This is the test that keeps :mod:`frozenaudit.core.contrast` honest after
    it was generalised out of the original analysis scripts. It recomputes all
    three arms for all five comparators from the frozen per-molecule
    predictions and compares against the published metrics file.

    It also checks the native-margin selection row by row against the frozen
    ``native_margin_matched_453`` column, because two selections can give the
    same balanced accuracy while disagreeing about which molecules they chose.
    """
    import pandas as pd

    from frozenaudit.core import contrast

    long_path = FROZEN / "abcb1_public_predictor_predictions_long_v1.csv"
    metrics_path = FROZEN / "abcb1_public_predictor_metrics_v1.csv"
    if not (long_path.exists() and metrics_path.exists()):
        pytest.skip(f"frozen prediction/metric files not available in {FROZEN}")

    long = pd.read_csv(long_path)
    pub = pd.read_csv(metrics_path)

    def published(cid, population):
        hit = pub[(pub.comparator_id == cid) & (pub.population == population)]
        return float(hit.balanced_accuracy.iloc[0]) if len(hit) else None

    arms_to_population = {
        "full": "all_identity_clean",
        "selected": "v0_4_accepted_mask",
        "native_margin_matched": "native_margin_matched_453",
    }

    seen = 0
    for cid, g in long.groupby("comparator_id"):
        g = g.sort_values("external_row_id")
        threshold = float(g.binary_threshold.iloc[0])
        out = contrast.coverage_matched_contrast(
            scores=g.score.values,
            labels=g.observed_er2_label.values,
            threshold=threshold,
            selector_accepted=g.v0_4_accepted.values,
            predictions=g.predicted_label.values,
            selector_name="v0_4",
        )
        for arm, population in arms_to_population.items():
            expected = published(cid, population)
            if expected is None:
                continue
            got = out["arms"][arm]["metrics"]["balanced_accuracy"]
            assert got == pytest.approx(expected, abs=1e-12), (
                f"{cid} / {population}: contrast gives {got}, "
                f"published value is {expected}"
            )
            seen += 1

        mine = contrast.select_by_native_margin(
            g.score.values, threshold, int(g.v0_4_accepted.sum()))
        assert (mine == g.native_margin_matched_453.values).all(), (
            f"{cid}: native-margin selection disagrees with the frozen column"
        )

    assert seen >= 15, f"only {seen} published values were checked"


def test_unparseable_query_abstains():
    """An unparseable structure is unknown, not distant: no number is invented."""
    pool = _pool()
    th = domain.d5_thresholds(domain.leave_one_connectivity_out_d5(pool))
    (row,) = domain.score_queries(["not a smiles"], pool, th)
    assert row["parsed"] is False
    assert row["d5"] is None and row["stratum"] is None


def test_gate_small_n_is_not_evaluable():
    """Too few rows gives not_evaluable, never fail."""
    g = gates.Gate("direction", "95% lower bound > 0", min_n=20,
                   rule=gates.ci_lower_above(0.0), requires=("ci95_low",))
    assert g.evaluate({"n": 7, "ci95_low": 0.31})["verdict"] == gates.NOT_EVALUABLE
    assert g.evaluate({"n": 62, "ci95_low": 0.31})["verdict"] == gates.PASS
    assert g.evaluate({"n": 62, "ci95_low": -0.02})["verdict"] == gates.FAIL


def test_summary_partial_is_not_pass():
    """One pass and two not_evaluable is reported as partial, not as a pass."""
    g = gates.Gate("g", "declared", min_n=20, rule=gates.point_above(0.0),
                   requires=("estimate",))
    res = [g.evaluate({"n": 30, "estimate": 0.2}),
           g.evaluate({"n": 7, "estimate": 0.2}),
           g.evaluate({"n": 7, "estimate": 0.2})]
    s = gates.summarise(res)
    assert s["overall"] == "partial" and s["counts"][gates.PASS] == 1
