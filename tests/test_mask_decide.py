"""The README's documented entry point must actually answer.

`mask.decide` is the first thing a reader runs, and until this file existed no
test called it: the suite was green while the documented call returned
``abstain`` with ``screener_error: AttributeError`` for every input, on any
interpreter whose scikit-learn predated the one the bundles were pickled with.
A silent abstention is the worst possible failure here, because abstaining *is*
a legitimate output of the protocol -- so a broken install looked like a
scientific result.

These tests skip, rather than fail, when the runtime cannot load the bundles:
that condition is reported by `mask.runtime_problem()` and is not a defect in
this package.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frozenaudit.instances.abcb1 import mask  # noqa: E402

pytestmark = pytest.mark.skipif(not mask.runtime_is_supported(),
                                reason=str(mask.runtime_problem()))

# Two small, unambiguous structures: paracetamol and nicotine. Nothing here
# depends on what the screener decides about them -- only that it decides.
QUERIES = ["CC(=O)Nc1ccc(O)cc1", "CN1CCC[C@H]1c1cccnc1"]


def test_decide_returns_a_decision_not_an_internal_error():
    """Every row carries a real three-state call, and no screener_error reason."""
    out = mask.decide(QUERIES)
    assert len(out) == len(QUERIES)
    for row in out:
        assert row["risk_call"] in mask.ANSWERED + ("abstain",), row
        assert not any(r.startswith("screener_error") for r in row["reasons"]), row
        assert row["accepted"] == (row["risk_call"] in mask.ANSWERED)


def test_unparseable_structure_abstains_without_raising():
    """A bad SMILES is unknown, not broken: abstain with the structural reason."""
    (row,) = mask.decide(["not a smiles"])
    assert row["accepted"] is False
    assert row["risk_call"] == "abstain"
    assert "unresolved_structure" in row["reasons"]


def test_internal_failure_raises_instead_of_abstaining():
    """A screener that raises must not be reported as an abstention."""
    bundles = mask.load_bundles(disclose=False)
    broken = dict(bundles)
    broken["selective"] = {}          # missing every key predict_one reads
    with pytest.raises(mask.ScreenerError):
        mask.decide(QUERIES[:1], broken)


def test_shipped_files_match_the_manifest():
    """The distributed bundles and reference CSV hash to their recorded values."""
    checked = mask.verify_shipped_files()
    assert len(checked) >= 4, checked
    assert all(len(h) == 64 for h in checked.values())
