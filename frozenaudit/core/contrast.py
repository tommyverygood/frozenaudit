"""Step 3 of the protocol: is the external selector better than the model's own confidence?

This is the comparison that decides whether an abstention layer earns its
place, and the easiest one to get wrong.

The wrong comparison
--------------------
Score a model on everything, then score it again on the subset your selector
accepted, and report the difference. That number is almost always positive and
almost always meaningless: any selector that removes rows will look good,
because the rows a model gets wrong are correlated with the rows any
reasonable selector drops. Reporting reliability without coverage is the same
error in a different coat.

The comparison this module makes
--------------------------------
Four arms, all at explicitly reported coverage:

``full``
    every row, no selection -- the floor.
``selected``
    the rows the external selector accepted.
``selected_complement``
    the rows it rejected. If the selector is doing anything, this arm should be
    worse than ``full``; if it is roughly equal, the selector is thinning the
    set without concentrating the errors.
``native_margin_matched``
    the same *number* of rows as ``selected``, chosen instead by the model's
    own distance from its decision threshold. This is the arm that matters.

A selector only earns its place by beating ``native_margin_matched``, because
that arm is free -- it needs no extra model, no reference panel, nothing but
the scores you already have. In the accompanying paper this arm is what
reversed the conclusion for the one endpoint-aligned, well-calibrated
comparator: the external mask improved on ``full`` there too, and still lost
to the model's own margin by 0.108 balanced accuracy.

The ``not_evaluable`` discipline from :mod:`frozenaudit.core.gates` applies
here as well: an arm holding fewer than ``min_n`` rows reports ``None`` for
every metric, never a number. A balanced accuracy computed on nine molecules
is not a small-sample estimate of anything, and putting it in a table invites
a reader to compare it with one computed on nine hundred.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = ["Arm", "margin", "select_by_native_margin", "coverage_matched_contrast",
           "METRICS", "balanced_accuracy", "mcc"]


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Mean of sensitivity and specificity; None if either class is absent.

    Returning None rather than 0.5 for a single-class arm is deliberate. With
    one class present, sensitivity or specificity is undefined, and 0.5 would
    read as "chance performance" when the truth is "this arm cannot say".
    """
    pos, neg = y_true == 1, y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    sens = float((y_pred[pos] == 1).mean())
    spec = float((y_pred[neg] == 0).mean())
    return (sens + spec) / 2.0


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Matthews correlation coefficient; None when the denominator vanishes."""
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return None
    return float((tp * tn - fp * fn) / denom)


METRICS: Mapping[str, Callable[[np.ndarray, np.ndarray], float | None]] = {
    "balanced_accuracy": balanced_accuracy,
    "mcc": mcc,
}


@dataclass
class Arm:
    """One arm of the contrast, with its coverage attached to its metrics.

    ``coverage`` is a fraction of the *full* set, not of anything else, so the
    four arms of one contrast are directly comparable.
    """

    name: str
    n: int
    coverage: float
    positive_prevalence: float | None
    metrics: dict
    evaluable: bool
    why: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def margin(scores: Sequence[float], threshold: float) -> np.ndarray:
    """Absolute distance of each score from the decision threshold.

    This is the model's own confidence ordering, and the only baseline
    selector that costs nothing to apply.
    """
    return np.abs(np.asarray(scores, dtype=float) - float(threshold))


def select_by_native_margin(scores: Sequence[float], threshold: float,
                            n: int) -> np.ndarray:
    """Boolean mask of the ``n`` rows furthest from ``threshold``.

    Ties at the cut are broken by original row order (stable sort), so the
    selection is reproducible. NaN scores sort last and are therefore never
    selected: a missing score is not confidence.
    """
    m = margin(scores, threshold)
    m = np.where(np.isnan(m), -np.inf, m)
    n = int(min(max(n, 0), m.size))
    # negate for descending order; 'stable' makes ties fall in row order
    order = np.argsort(-m, kind="stable")[:n]
    out = np.zeros(m.size, dtype=bool)
    out[order] = True
    return out


def _arm(name: str, y_true: np.ndarray, y_pred: np.ndarray, total: int,
         metric_names: Sequence[str], min_n: int) -> Arm:
    n = int(y_true.size)
    coverage = n / total if total else 0.0
    if n < min_n:
        return Arm(name=name, n=n, coverage=coverage, positive_prevalence=None,
                   metrics={k: None for k in metric_names}, evaluable=False,
                   why=f"n = {n} below the declared minimum of {min_n}")
    prev = float((y_true == 1).mean())
    values = {k: METRICS[k](y_true, y_pred) for k in metric_names}
    undefined = [k for k, v in values.items() if v is None]
    return Arm(name=name, n=n, coverage=coverage, positive_prevalence=prev,
               metrics=values, evaluable=not undefined,
               why="" if not undefined
                   else f"undefined for this arm: {', '.join(sorted(undefined))} "
                        f"(positive prevalence {prev:.3f})")


def coverage_matched_contrast(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    threshold: float,
    selector_accepted: Sequence[bool],
    predictions: Sequence[int] | None = None,
    metrics: Sequence[str] = ("balanced_accuracy", "mcc"),
    min_n: int = 30,
    selector_name: str = "selector",
) -> dict:
    """Run the four-arm contrast.

    Parameters
    ----------
    scores
        The audited model's continuous output, one per row. Used both for the
        binary call (via ``threshold``) and for the native-margin arm.
    labels
        Ground truth, 0/1. Rows whose label is NaN/None are dropped from every
        arm before anything is computed, and the drop is reported.
    threshold
        The model's decision threshold, declared before scoring.
    selector_accepted
        Boolean per row: did the external selector agree to answer here?
    predictions
        Optional explicit binary calls. When omitted, ``scores >= threshold``
        is used, which is what "the model's own call" means in this context.
    min_n
        Below this many rows an arm is ``not_evaluable`` and reports no metrics.
    selector_name
        Label for the selector arms in the returned dict.

    Returns
    -------
    dict
        ``{"arms": {...}, "deltas": {...}, "reading": str, "dropped": int,
        "n_total": int}``. ``deltas`` carries the selector minus each baseline
        for every requested metric; ``reading`` states in words which selector
        won, including the case where the model's own margin won.
    """
    for name in metrics:
        if name not in METRICS:
            raise ValueError(f"unknown metric {name!r}; available: {sorted(METRICS)}")

    s = np.asarray(scores, dtype=float)
    y = np.asarray([np.nan if v is None else v for v in labels], dtype=float)
    acc = np.asarray(selector_accepted, dtype=bool)
    if not (s.size == y.size == acc.size):
        raise ValueError(f"length mismatch: scores {s.size}, labels {y.size}, "
                         f"selector_accepted {acc.size}")

    keep = ~np.isnan(y) & ~np.isnan(s)
    dropped = int((~keep).sum())
    s, acc = s[keep], acc[keep]
    y = y[keep].astype(int)
    if predictions is None:
        pred = (s >= threshold).astype(int)
    else:
        pred = np.asarray(predictions)[keep].astype(int)

    total = int(y.size)
    if total == 0:
        raise ValueError("no rows left after dropping missing scores/labels")

    n_sel = int(acc.sum())
    native = select_by_native_margin(s, threshold, n_sel)

    arms = {
        "full": _arm("full", y, pred, total, metrics, min_n),
        "selected": _arm(f"{selector_name}_accepted", y[acc], pred[acc], total,
                         metrics, min_n),
        "selected_complement": _arm(f"{selector_name}_rejected", y[~acc],
                                    pred[~acc], total, metrics, min_n),
        "native_margin_matched": _arm(f"native_margin_matched_{n_sel}",
                                      y[native], pred[native], total, metrics,
                                      min_n),
    }

    deltas, verdicts = {}, []
    for k in metrics:
        sel = arms["selected"].metrics.get(k)
        base_full = arms["full"].metrics.get(k)
        base_native = arms["native_margin_matched"].metrics.get(k)
        deltas[k] = {
            "vs_full": None if sel is None or base_full is None else sel - base_full,
            "vs_native_margin": (None if sel is None or base_native is None
                                 else sel - base_native),
        }
        d = deltas[k]["vs_native_margin"]
        if d is None:
            verdicts.append(f"{k}: not evaluable against the native margin")
        elif d > 0:
            verdicts.append(f"{k}: {selector_name} beats the model's own margin "
                            f"by {d:+.3f} at {arms['selected'].coverage:.1%} coverage")
        else:
            verdicts.append(f"{k}: the model's own margin beats {selector_name} "
                            f"by {-d:+.3f} at {arms['selected'].coverage:.1%} "
                            "coverage -- the external selector is not warranted "
                            "on this model")

    return {"n_total": total, "dropped": dropped, "n_selected": n_sel,
            "coverage": n_sel / total,
            "arms": {k: v.as_dict() for k, v in arms.items()},
            "deltas": deltas, "reading": "; ".join(verdicts)}
