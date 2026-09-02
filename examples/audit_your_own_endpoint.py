"""Audit any binary predictor with frozenaudit.core -- no chemistry involved.

This is the general path: the one you follow when the endpoint is yours rather
than ABCB1. It needs three columns and nothing else:

    score     the model's continuous output, higher = more likely positive
    observed  ground truth, 1 / 0
    accepted  optional -- your selector's accept/abstain, if you have one

Run the built-in synthetic demo::

    python examples/audit_your_own_endpoint.py

Or point it at your own file::

    python examples/audit_your_own_endpoint.py mypreds.csv --threshold 0.5

The demo deliberately uses a *bad* selector -- one that keeps the rows nearest
the decision threshold -- so that the report comes back saying the model's own
margin would have done better. That is the outcome this protocol exists to be
able to report, and a demo that only showed the flattering case would be
teaching the wrong lesson.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frozenaudit.core import Audit, gates  # noqa: E402


def synthetic(n: int = 800, seed: int = 0):
    """Scores correlated with labels, plus a selector that keeps the hard rows."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    scores = np.clip(0.5 + 0.25 * (labels * 2 - 1) + rng.normal(0, 0.22, n), 0, 1)
    nearest = np.argsort(np.abs(scores - 0.5))[: n // 4]
    accepted = np.zeros(n, dtype=bool)
    accepted[nearest] = True
    return scores, labels, accepted


def load(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    for col in ("score", "observed"):
        if col not in df.columns:
            raise SystemExit(f"{path} needs a {col!r} column; found {list(df.columns)}")
    scores = df["score"].to_numpy(dtype=float)
    labels = df["observed"].to_numpy()
    if "accepted" in df.columns:
        accepted = df["accepted"].to_numpy(dtype=bool)
    else:
        # No selector supplied: audit the model against its own margin at a
        # coverage you choose. 25% is a placeholder, not a recommendation --
        # pick it from what your downstream use can afford to abstain on.
        keep = max(1, int(0.25 * len(df)))
        margin = np.abs(scores - 0.5)
        accepted = np.zeros(len(df), dtype=bool)
        accepted[np.argsort(-margin, kind="stable")[:keep]] = True
        print("no 'accepted' column: using the model's own top-25% margin as the "
              "selector, which makes the contrast a self-consistency check\n",
              file=sys.stderr)
    return scores, labels, accepted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictions", nargs="?", type=Path,
                    help="CSV with score, observed and optionally accepted; "
                         "omit for the synthetic demo")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="the model's decision threshold (default 0.5)")
    ap.add_argument("--min-n", type=int, default=30,
                    help="below this many rows an arm is not_evaluable")
    ap.add_argument("--name", default="my_endpoint_v1")
    args = ap.parse_args()

    if args.predictions is None:
        scores, labels, accepted = synthetic()
        print("(synthetic demo: 800 rows, selector deliberately poor)\n",
              file=sys.stderr)
    else:
        scores, labels, accepted = load(args.predictions)

    audit = Audit(args.name, gates=[
        gates.Gate(
            name="selector_beats_own_margin",
            declared="the selector's balanced accuracy must exceed the "
                     "model's own margin-matched arm at equal coverage",
            min_n=args.min_n,
            rule=gates.point_above(0.0),
            requires=("estimate",),
        ),
    ])

    # Step 1, before anything is scored.
    lock = audit.freeze({"threshold": args.threshold, "min_n": args.min_n,
                         "metric": "balanced_accuracy"},
                        note=f"audit of {args.predictions or 'synthetic demo'}")

    # Step 3.
    out = audit.contrast(scores=scores, labels=labels, threshold=args.threshold,
                         selector_accepted=accepted, min_n=args.min_n)

    # Step 4: feed the contrast's own delta into the declared gate.
    delta = out["deltas"]["balanced_accuracy"]["vs_native_margin"]
    audit.run_gates({"selector_beats_own_margin": {
        "n": out["n_selected"], "estimate": delta}})

    report = audit.report()

    print(f"freeze     {lock.short}  ({lock.label})")
    print(f"coverage   {out['coverage']:.1%}  ({out['n_selected']} of "
          f"{out['n_total']} rows)")
    print("\narms (balanced accuracy):")
    for key, arm in out["arms"].items():
        ba = arm["metrics"]["balanced_accuracy"]
        shown = "not evaluable" if ba is None else f"{ba:.4f}"
        print(f"  {arm['name']:<34s} n={arm['n']:>5d}  {shown}"
              + (f"   [{arm['why']}]" if arm["why"] else ""))
    print(f"\nreading    {out['reading']}")
    print(f"overall    {report['overall']}")
    for caveat in report["caveats"]:
        print(f"  caveat   {caveat}")

    Path("audit_report.json").write_text(json.dumps(report, indent=2,
                                                    default=str))
    print("\nwrote audit_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
