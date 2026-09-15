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

Runtime
-------
The three bundles were pickled with the versions in ``requirements-lock.txt``
(scikit-learn 1.9.0). An older scikit-learn unpickles them without complaint
and then fails inside ``predict_proba``, so :func:`load_bundles` refuses to
load on an unsupported runtime instead of letting the failure surface later as
an abstention. See :func:`runtime_problem`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import joblib

_HERE = Path(__file__).resolve().parent
# model/ ships inside the package, so a non-editable install keeps it importable.
_MODEL_DIR = _HERE / "model"

SELECTIVE = _MODEL_DIR / "ABCB1_selective_uncertainty_v0_4_dev.joblib"
PRIMARY = _MODEL_DIR / "ABCB1_screener_v0_1.joblib"
DIRECT = _MODEL_DIR / "ABCB1_assay_aware_candidate_v0_3_dev.joblib"
MANIFEST = _MODEL_DIR / "manifest.json"

ANSWERED = ("high_ABCB1_efflux_risk", "low_ABCB1_efflux_risk")

# The bundles were fitted and pickled under this scikit-learn. Unpickling under
# an older one raises InconsistentVersionWarning and then breaks at predict
# time (1.9 dropped attributes that 1.3-era estimators expect to find), so the
# floor is a hard requirement rather than a recommendation.
MIN_SKLEARN = (1, 9)


class ScreenerError(RuntimeError):
    """The shipped screener failed on an input for a reason that is not a decision.

    Raised rather than converted into an abstention: an abstention is a
    reported outcome of the protocol, and a library that returns one after an
    internal failure makes a broken install indistinguishable from chemistry
    outside the reference pool.
    """


def runtime_problem() -> str | None:
    """Describe why this interpreter cannot load the bundles, or return None.

    Kept separate from :func:`require_runtime` so that callers and tests can
    branch on the condition without catching an exception.
    """
    try:
        import sklearn
    except ImportError:
        return ("scikit-learn is not installed; the ABCB1 instance needs it. "
                'Install the extra: pip install "frozenaudit[abcb1]"')
    parts = []
    for chunk in str(sklearn.__version__).split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    found = tuple(parts)
    if found < MIN_SKLEARN:
        want = ".".join(str(n) for n in MIN_SKLEARN)
        return (f"the shipped bundles were pickled with scikit-learn "
                f"{want}.x but this interpreter has {sklearn.__version__}. "
                f"They would unpickle and then fail at predict time. "
                f'Install a supported runtime: pip install "scikit-learn>={want}" '
                f"(see requirements-lock.txt for the full tested set)")
    return None


def runtime_is_supported() -> bool:
    """True when the bundles can be loaded on this interpreter."""
    return runtime_problem() is None


def require_runtime() -> None:
    """Raise :class:`ScreenerError` if the runtime cannot load the bundles."""
    problem = runtime_problem()
    if problem is not None:
        raise ScreenerError(problem)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_shipped_files(model_dir: Path = _MODEL_DIR) -> dict:
    """Check the shipped files against ``manifest.json``'s ``shipped_sha256``.

    The v0.4 bundle carries ``dependency_hashes`` for the other two, and the
    screener verifies those on load, but nothing covered the v0.4 bundle
    itself or the reference CSV -- a bundle cannot hash itself. The manifest
    closes that gap for the files as distributed.

    Returns a path -> sha256 mapping of what was checked. Raises
    :class:`ScreenerError` on the first mismatch, naming the file: a changed
    bundle is a different protocol, not a warning.
    """
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise ScreenerError(f"{manifest_path} is missing; the shipped files "
                            "cannot be verified")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get(
        "shipped_sha256", {})
    if not recorded:
        raise ScreenerError(f"{manifest_path} has no shipped_sha256 block; "
                            "this package predates integrity verification")
    checked = {}
    package_dir = model_dir.parent
    for rel, expected in recorded.items():
        path = package_dir / rel
        if not path.exists():
            raise ScreenerError(f"{rel} is recorded in manifest.json but is "
                                "not present; the package is incomplete")
        found = _sha256(path)
        if found != expected:
            raise ScreenerError(
                f"{rel} does not match manifest.json: expected "
                f"{expected[:16]}..., found {found[:16]}.... A modified bundle "
                "is a different protocol; do not report numbers from it as this "
                "instance's."
            )
        checked[rel] = found
    return checked


def load_bundles(selective: Path = SELECTIVE, primary: Path = PRIMARY,
                 direct: Path = DIRECT, disclose: bool = True,
                 verify: bool = True) -> dict:
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

    Loading goes through the shipped screener's own ``load_packages``, which
    verifies the v0.1 and v0.3 bundles against the hashes recorded inside the
    v0.4 bundle. ``verify=True`` additionally checks every shipped file against
    ``manifest.json``. Pass ``verify=False`` only when deliberately loading
    substituted bundles, and say so wherever you report the result.

    Raises
    ------
    ScreenerError
        If the runtime cannot load the bundles, or a shipped file does not
        match its recorded hash.
    """
    require_runtime()
    if verify and (selective, primary, direct) == (SELECTIVE, PRIMARY, DIRECT):
        verify_shipped_files()
    mod = _screener_module()
    # load_packages checks sha256 of primary and direct against the hashes
    # frozen inside the selective bundle, and raises on mismatch.
    sel, pri, dir_ = mod.load_packages(selective, primary, direct)
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
    (empty when accepted).

    An unparseable structure abstains with ``unresolved_structure`` rather than
    raising -- that comes from the screener's own return value, not from
    catching an exception here. Anything that does raise is an internal
    failure and is re-raised as :class:`ScreenerError`, because an abstention
    returned for a broken install would be indistinguishable from an
    abstention returned for chemistry outside the reference pool.
    """
    b = bundles if bundles is not None else load_bundles()
    mod = _screener_module()
    out = []
    for smiles in smiles_list:
        try:
            res = mod.predict_one(smiles, b["selective"], b["primary"], b["direct"])
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise ScreenerError(
                f"the shipped screener failed on {smiles!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
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
