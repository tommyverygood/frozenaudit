"""The endpoint-agnostic core: the protocol, with no property and no molecule in it.

Nothing in this subpackage knows what ABCB1 is. The four modules below take a
reference set, some scores, some labels and some declared criteria, and they
would run the same way on a solubility model, a clearance model, or -- for
:mod:`.gates`, :mod:`.contrast` and :mod:`.freeze` -- on something that is not
chemistry at all.

* :mod:`.freeze`    hash a protocol description so drift is detectable
* :mod:`.domain`    structure-only distance to a reference pool, and strata
* :mod:`.contrast`  four-arm coverage-matched comparison against the model's
                    own confidence
* :mod:`.gates`     pre-declared criteria with a ``not_evaluable`` verdict
* :mod:`.protocol`  :class:`~.protocol.Audit`, which sequences the above and
                    records whether the freeze preceded the scoring

Dependency note
---------------
:mod:`.domain` is the only module here that needs RDKit, and it is imported
lazily -- ``from frozenaudit.core import gates, contrast, freeze`` works in an
environment with no chemistry stack installed. That is deliberate: a
non-chemical audit should not have to install a cheminformatics toolkit to use
the gate logic.

What is *not* here
------------------
Any reference panel, any model weights, any endpoint. Those live under
:mod:`frozenaudit.instances`. If you find yourself wanting to add a
property-specific default to this subpackage, that default belongs in an
instance instead.
"""

from __future__ import annotations

from . import contrast, freeze, gates  # noqa: F401
from .freeze import FreezeLock, FreezeMismatch, freeze_hash  # noqa: F401
from .gates import Gate, FAIL, NOT_EVALUABLE, PASS  # noqa: F401
from .protocol import Audit  # noqa: F401

__all__ = ["Audit", "FreezeLock", "FreezeMismatch", "Gate", "contrast",
           "freeze", "freeze_hash", "gates", "domain",
           "PASS", "FAIL", "NOT_EVALUABLE"]


def __getattr__(name: str):
    """Import :mod:`.domain` on first use so RDKit stays an optional dependency.

    ``importlib.import_module`` rather than ``from . import domain``: the
    latter re-enters this hook while resolving the submodule and recurses.
    """
    if name == "domain":
        import importlib

        module = importlib.import_module(f"{__name__}.domain")
        globals()["domain"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
