"""The development-artefact caveat must actually reach the user.

README promises that loading the bundles writes the artefacts' self-declared
status to stderr. These tests hold that promise to the code, so the caveat
cannot quietly disappear in a later refactor.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frozenaudit.instances.abcb1 import mask  # noqa: E402

EXPECTED = ("development_selective_risk_screener_not_future_source_validated",
            "development_research_screener_gate_failed",
            "single_source_internal_development_candidate_not_externally_validated")


def test_import_is_silent(capsys):
    """Importing the package prints nothing: side effects on import are rude."""
    import importlib

    import frozenaudit
    capsys.readouterr()
    importlib.reload(frozenaudit)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_load_bundles_discloses_to_stderr(capsys):
    """Every declared status string reaches stderr on a default load."""
    mask.load_bundles()
    err = capsys.readouterr().err
    for token in EXPECTED:
        assert token in err, f"{token!r} was not disclosed"


def test_disclose_false_is_silent(capsys):
    """The disclosure can be silenced explicitly, and only explicitly."""
    mask.load_bundles(disclose=False)
    assert capsys.readouterr().err == ""


def test_declared_matches_bundle_contents():
    """The disclosed strings are read from the bundles, not hardcoded here."""
    declared = mask.load_bundles(disclose=False)["declared"]
    assert declared["v0.4_status"] == EXPECTED[0]
    assert declared["v0.1_claim_status"] == EXPECTED[1]
    assert declared["v0.3_status"] == EXPECTED[2]
    assert declared["v0.4_claim_boundary"]
