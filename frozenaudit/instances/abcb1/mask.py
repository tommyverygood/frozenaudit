"""The frozen accept/abstain mask, and how to put it in front of your model.

What this is
------------
The mask is *not* a model-agnostic applicability-domain filter. Its first
condition is that the shipped v0.4 screener's own calibrated probability be
extreme (>= 0.70 or <= 0.35), and two later checks read that screener's
bootstrap interval and its v0.3 companion. So "accept" means:

    the shipped screener was confident here, and the structure is supported
    by the reference pool and inside the frozen physicochemical envelope.

Using it on your own model therefore means: run this screener alongside your
model, and keep only your model's predictions on the molecules this screener
would have answered. That is exactly what the accompanying paper did.

When it helps, and when it does not
-----------------------------------
Measured on one external set of 2,624 identity-clean molecules, at the
17.3% coverage the mask itself selects, against each model's *own*
threshold-distance margin taken at the same coverage:

    Deep-PK P-gp substrate            0.529 -> 0.628   (+0.098)
    this project's v0.4 score          0.637 -> 0.735   (+0.098)
    ADMET-AI P-gp inhibition           0.516 -> 0.561   (+0.045)
    AZ GNN-MTL NIH-MDCK-ER             0.703 -> 0.736   (+0.033)
    AZ GNN-MTL MDCK-ER                 0.902 -> 0.794   (-0.108)

(balanced accuracy; see the paper for intervals and the coverage-matching
construction.)

Read the last row before using this. On a well-calibrated, endpoint-aligned
score, the model's own distance from its decision threshold is the better
selector and this mask makes things worse. The mask earns its place when the
base score is poorly calibrated or its endpoint does not match the question
being asked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib

_HERE = Path(__file__).resolve().parent
# frozenaudit/instances/abcb1/ -> repo root -> model/
_MODEL_DIR = _HERE.parent.parent.parent / "model"

SELECTIVE = _MODEL_DIR / "ABCB1_selective_uncertainty_v0_4_dev.joblib"
PRIMARY = _MODEL_DIR / "ABCB1_screener_v0_1.joblib"
DIRECT = _MODEL_DIR / "ABCB1_assay_aware_candidate_v0_3_dev.joblib"

ANSWERED = ("high_ABCB1_efflux_risk", "low_ABCB1_efflux_risk")


def _screener_module():
    """Import the shipped screener CLI as a module.

    The scoring code is the file that produced the paper's numbers; it is
    imported rather than reimplemented so that a mask computed here cannot
    drift from the one reported.
    """
    path = _HERE / "_screener_cli.py"
    spec = importlib.util.spec_from_file_location("_frozenaudit_screener", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_bundles(selective: Path = SELECTIVE, primary: Path = PRIMARY,
                 direct: Path = DIRECT, disclose: bool = True) -> dict:
    """Load the three frozen bundles and report what they say about themselves.

    The ``status`` and ``claim_boundary`` strings are returned rather than
    hidden: every one of the three declares itself a development artefact, and
    callers should surface that wherever results are shown.

    With ``disclose=True`` (the default) those declarations are written to
    stderr on every call, so a user who never opens the README still sees them
    before any number appears. The disclosure is emitted here rather than at
    import time because a library that prints when imported is a nuisance in
    other people's pipelines; pass ``disclose=False`` to silence it once you
    are reporting the strings yourself.
    """
    sel, pri, dir_ = joblib.load(selective), joblib.load(primary), joblib.load(direct)
    declared = {
        "v0.4_status": sel.get("status"),
        "v0.4_claim_boundary": sel.get("claim_boundary"),
        "v0.1_claim_status": pri.get("claim_status"),
        "v0.1_claim_boundary": pri.get("claim_boundary"),
        "v0.3_status": dir_.get("status"),
    }
    if disclose:
        print("frozenaudit: the bundled artefacts declare themselves as follows. "
              "Report this wherever you report numbers derived from them.",
              file=sys.stderr)
        for key, value in declared.items():
            print(f"  {key}: {value}", file=sys.stderr)
        print(file=sys.stderr)
    return {"selective": sel, "primary": pri, "direct": dir_, "declared": declared}


def decide(smiles_list, bundles: dict | None = None) -> list[dict]:
    """accept/abstain for each SMILES, with the machine-readable reason.

    Returns one dict per input: ``accepted`` (bool), ``risk_call`` (the
    screener's own three-state output), ``probability``, and ``reasons``
    (empty when accepted). An unparseable structure abstains with a reason
    rather than raising.
    """
    b = bundles if bundles is not None else load_bundles()
    mod = _screener_module()
    out = []
    for smiles in smiles_list:
        try:
            res = mod.predict_one(smiles, b["selective"], b["primary"], b["direct"])
        except Exception as exc:                     # noqa: BLE001
            out.append({"smiles": smiles, "accepted": False, "risk_call": "abstain",
                        "probability": None,
                        "reasons": [f"screener_error: {type(exc).__name__}"]})
            continue
        call = res.get("decision") or res.get("risk_call") or res.get("final_decision_state")
        reasons = res.get("abstention_reasons") or res.get("reasons") or []
        out.append({"smiles": smiles, "accepted": call in ANSWERED, "risk_call": call,
                    "probability": res.get("primary_calibrated_probability")
                    or res.get("probability"),
                    "reasons": list(reasons)})
    return out


def apply_to(your_predictions, mask_decisions) -> list[dict]:
    """Keep your model's rows where the mask accepted, tagging the rest.

    ``your_predictions`` and ``mask_decisions`` must be aligned row-wise. This
    function does not compute a metric: it returns the filtered rows so you can
    score them with whatever measure your endpoint calls for, and report the
    coverage alongside -- a reliability number without its coverage is not
    comparable to anything.
    """
    if len(your_predictions) != len(mask_decisions):
        raise ValueError(
            f"length mismatch: {len(your_predictions)} predictions vs "
            f"{len(mask_decisions)} mask decisions"
        )
    return [{**p, "mask_accepted": m["accepted"], "mask_risk_call": m["risk_call"],
             "mask_reasons": m["reasons"]}
            for p, m in zip(your_predictions, mask_decisions)]
