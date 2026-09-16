"""EffluxScope -- a computational framework for evaluating ABCB1 efflux predictors
under chemical-space shift.

The Python package name ``frozenaudit`` is retained for compatibility.

The package is in two halves, and the split is the whole design:

:mod:`frozenaudit.core`
    The evaluation workflow. Endpoint-agnostic: record the rules, stratify by
    distance to a reference set, contrast against the model's own confidence at matched
    coverage, evaluate pre-declared gates that can return ``not_evaluable``,
    report what was and was not established. No property, no panel, no weights.

:mod:`frozenaudit.instances`
    Worked instances. One ships: :mod:`frozenaudit.instances.abcb1`, the
    ABCB1/P-glycoprotein efflux-risk evaluation from the accompanying paper,
    complete with its reference panel and its three fixed model bundles.

Two things this package is not
------------------------------
**It is not a property predictor.** The bundles under ``frozenaudit/instances/abcb1/model/`` are objects
of evaluation in the accompanying paper, not products of it. Every one of them
declares itself a development artefact, and the strict blind-pool gate on the
binary substrate call came back ``fail``. They ship so that the evaluation is
reproducible, and :func:`frozenaudit.instances.abcb1.mask.load_bundles` prints
their self-declarations to stderr on every call for that reason. If you want
an ABCB1 substrate prediction, a better free option exists -- see the README.

**It is not validated across endpoints.** The transfer evidence in the paper
is across five *models* on one *endpoint*. The core is written to be endpoint
-agnostic and its tests exercise it on synthetic non-chemical data, but
"this code does not hard-code an endpoint" is a statement about the code, not
evidence that the framework's conclusions carry to another property. Running it
on a second endpoint is open work; :mod:`frozenaudit.instances` documents the
interface for doing so.

Quick start
-----------
The general path, with your own model and your own reference set::

    from frozenaudit.core import Audit, Gate, gates

    audit = Audit("my_endpoint_v1", pool=my_pool, gates=[
        Gate("direction", "95% lower bound above zero", min_n=30,
             rule=gates.ci_lower_above(0.0), requires=("ci95_low",)),
    ])
    audit.freeze({"threshold": 0.5, "metric": "balanced_accuracy"})
    audit.stratify(query_smiles)
    audit.contrast(scores=scores, labels=labels, threshold=0.5,
                   selector_accepted=accepted)
    audit.run_gates({"direction": {"n": 512, "ci95_low": 0.11}})
    report = audit.report()

The ABCB1 instance, reproducing the paper's selector::

    from frozenaudit.instances.abcb1 import load_pool, mask

    decisions = mask.decide(smiles_list)
    tagged = mask.apply_to(my_predictions, decisions)
"""

from __future__ import annotations

__version__ = "0.2.0"

from . import core  # noqa: F401
from .core import Audit, FreezeLock, Gate  # noqa: F401

__all__ = ["core", "instances", "Audit", "Gate", "FreezeLock", "__version__"]


def __getattr__(name: str):
    """Lazily expose :mod:`frozenaudit.instances`.

    Instances carry heavy, endpoint-specific baggage (RDKit, joblib, model
    bundles on disk). Importing the top-level package should not drag any of
    that in for someone who only wants the core.
    """
    if name == "instances":
        import importlib

        module = importlib.import_module(f"{__name__}.instances")
        globals()["instances"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
