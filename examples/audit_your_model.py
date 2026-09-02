#!/usr/bin/env python3
"""Audit your own efflux-risk predictor: the workflow from the paper, end to end.

Input: a CSV with one row per molecule and three columns

    smiles     standardized structure
    score      your model's output (higher = more likely a substrate / high risk)
    observed   the measured binary outcome, 1 = high efflux, 0 = low

Output: a table comparing three selections at the *same* coverage --

    1. no selection            your score on every molecule
    2. your own margin         the rows where your score is furthest from its
                               own decision threshold
    3. the frozen mask         the rows where the shipped screener would answer

Comparing (3) against (1) alone is not a fair test: any selection of confident
rows looks better than the whole set. (2) is the control that makes the
comparison mean something, and it is the comparison the paper reports.

    python examples/audit_your_model.py predictions.csv
    python examples/audit_your_model.py predictions.csv --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frozenaudit.instances.abcb1 import mask  # noqa: E402


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Mean of sensitivity and specificity; None when a class is absent.

    Returning None rather than 0.5 matters: a selection that kept only
    positives has no measurable specificity, and reporting 0.5 there would
    silently invent a number.
    """
    pos, neg = y_true == 1, y_true == 0
    if not pos.any() or not neg.any():
        return None
    sens = float((y_pred[pos] == 1).mean())
    spec = float((y_pred[neg] == 0).mean())
    return 0.5 * (sens + spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="your model's decision threshold (default 0.5)")
    ap.add_argument("--smiles-column", default="smiles")
    ap.add_argument("--score-column", default="score")
    ap.add_argument("--observed-column", default="observed")
    ap.add_argument("--output", type=Path, default=None,
                    help="write the per-row audit here")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    for col in (args.smiles_column, args.score_column, args.observed_column):
        if col not in df.columns:
            ap.error(f"column {col!r} not found; columns are {list(df.columns)}")

    y = df[args.observed_column].astype(int).to_numpy()
    s = df[args.score_column].astype(float).to_numpy()
    call = (s >= args.threshold).astype(int)

    # load_bundles writes the artefacts' self-declared status to stderr.
    bundles = mask.load_bundles()

    decisions = mask.decide(df[args.smiles_column].tolist(), bundles)
    accepted = np.array([d["accepted"] for d in decisions])
    n_acc = int(accepted.sum())
    if n_acc == 0:
        print("The frozen mask accepted no molecule in this set, so no "
              "coverage-matched comparison is possible. This is a real outcome "
              "for chemistry far from the reference pool, not an error.")
        return 1

    # Coverage-matched control: your own margin, same number of rows.
    margin_order = np.argsort(-np.abs(s - args.threshold))
    own = np.zeros(len(df), dtype=bool)
    own[margin_order[:n_acc]] = True

    rows = []
    for label, sel in (("no selection", np.ones(len(df), dtype=bool)),
                       ("your own margin", own),
                       ("frozen mask", accepted)):
        ba = balanced_accuracy(y[sel], call[sel])
        rows.append({"selection": label, "n": int(sel.sum()),
                     "coverage": round(float(sel.mean()), 4),
                     "balanced_accuracy": None if ba is None else round(ba, 4)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    # A one-sided score distribution makes the margin control degenerate: the
    # rows furthest from the threshold all sit on the same side, so only one
    # class is ever predicted and balanced accuracy collapses to 0.5. Say so,
    # because otherwise the mask looks better than the comparison supports.
    if len(np.unique(call[own])) == 1:
        print(f"\nNote: within the margin-matched control every row is predicted "
              f"{int(call[own][0])}, so its balanced accuracy is 0.5 by "
              f"construction rather than by measurement. Your score spans "
              f"[{s.min():.3f}, {s.max():.3f}] around a threshold of "
              f"{args.threshold}; a one-sided spread does this. Treat the "
              f"comparison below as indicative only, and prefer a margin "
              f"definition that draws from both sides.")

    ba_own = out.loc[out.selection == "your own margin", "balanced_accuracy"].iloc[0]
    ba_mask = out.loc[out.selection == "frozen mask", "balanced_accuracy"].iloc[0]
    if ba_own is not None and ba_mask is not None:
        d = ba_mask - ba_own
        print(f"\nAt matched coverage the frozen mask is {d:+.4f} against your own "
              f"margin.")
        if abs(d) < 1e-12:
            print("The two selections performed identically here, which on a "
                  "small set usually means neither ordered the errors: check "
                  "how many rows each kept before reading anything into it.")
        elif d < 0:
            print("A negative value is the expected outcome for a "
                  "well-calibrated, endpoint-aligned score: use your own "
                  "margin instead.")
        else:
            print("A positive value is the case the mask is for: a base score "
                  "whose own probability scale does not order its errors well.")
        print("This is a point estimate on one set. Resample before reporting it.")

    if args.output:
        audit = mask.apply_to(df.to_dict("records"), decisions)
        pd.DataFrame(audit).to_csv(args.output, index=False)
        print(f"\nper-row audit written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
