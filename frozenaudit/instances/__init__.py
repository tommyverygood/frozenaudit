"""Worked instances of the protocol. One ships; the interface is documented here.

An *instance* is everything :mod:`frozenaudit.core` deliberately does not
contain: a specific endpoint, a reference set for it, the model or models being
audited, and the criteria that were declared for that particular question.

Shipped
-------
:mod:`frozenaudit.instances.abcb1`
    ABCB1/P-glycoprotein efflux risk, as audited in the accompanying paper.
    477-member joint reference pool, three frozen bundles, and the frozen
    accept/abstain mask.

Adding your own
---------------
There is no plugin registry to register with and no base class to subclass --
an instance is a module that exposes three things by convention:

``load_pool() -> list[dict]``
    The reference set, in the form :func:`frozenaudit.core.domain.build_pool`
    returns: one dict per member with ``reference_id``, ``reference_name``,
    ``standardized_smiles`` and ``standardized_inchikey``. For most audits this
    is the audited model's own training set. If you do not know what a released
    model was trained on, say so in your report -- do not substitute a
    convenient public set and call the result an applicability domain.

``declared_gates() -> list[Gate]``
    The criteria, written before scoring. Keep the ``declared`` string verbatim
    as you first wrote it; :class:`~frozenaudit.core.protocol.Audit` hashes it,
    so an edit after the fact is detectable.

``DECLARED: dict``
    Whatever the instance must disclose about itself -- model status strings,
    claim boundaries, licence and attribution for any redistributed reference
    data. The ABCB1 instance's ``DECLARED`` is a useful template: it carries
    the three status strings and the blind-gate ``fail`` verdict, because an
    instance that hides its own negative results is worse than no instance.

A fourth is optional:

``mask`` / any selector module
    Only if your instance has an external selector to contribute. Most will
    not, and should not invent one -- the honest default selector is the
    audited model's own margin, which :mod:`frozenaudit.core.contrast`
    computes for free as the baseline arm.

Licence caution for instance authors
------------------------------------
The moment an instance ships a reference panel, the repository is
redistributing someone's data and the code licence stops being the whole
story. The ABCB1 instance's panel is CC BY-SA 3.0 for the structures and
CC BY 4.0 for the labels, both requiring attribution, one requiring
share-alike. Check what you are allowed to ship before you ship it, and record
the answer in ``DECLARED``.
"""

from __future__ import annotations

__all__ = ["abcb1", "AVAILABLE"]

AVAILABLE = {
    "abcb1": "ABCB1/P-glycoprotein efflux risk (accompanying paper; "
             "477-member joint pool, three frozen development bundles)",
}


def __getattr__(name: str):
    if name == "abcb1":
        import importlib

        module = importlib.import_module(f"{__name__}.abcb1")
        globals()["abcb1"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
