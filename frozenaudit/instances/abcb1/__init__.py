"""The ABCB1/P-glycoprotein efflux-risk instance from the accompanying paper.

This module supplies the three things :mod:`frozenaudit.instances` documents --
:func:`load_pool`, :func:`declared_gates`, :data:`DECLARED` -- plus the frozen
selector in :mod:`.mask`.

The reference pool
------------------
477 members: the 443-row primary reference concatenated with the 63-row direct
MDR1 panel, de-duplicated on InChIKey connectivity block so stereoisomers
collapse to one member. 443 + 63 = 506 rows in, 477 out; the 29 collapsed rows
are stereoisomer pairs, not dropped data.

:func:`load_pool` is the same construction the reconciliation test uses, and
the reason it lives here rather than in the test file is that a user auditing
their own model needs the pool, and copying it out of a test is how protocols
drift.

Provenance and licence of what this instance redistributes
----------------------------------------------------------
Every one of the 443 primary rows traces to a single article: Daood et al.
2025, *Mol Pharm*, DOI ``10.1021/acs.molpharmaceut.5c01065`` (PMC12587445),
CC BY 4.0. Their ``evidence_tier`` is uniformly
``B_author_curated_no_row_primary_assay`` -- author-curated labels, not
per-row primary assay measurements. The structures carry ChEMBL's CC BY-SA 3.0
share-alike obligation, which the MIT licence on the code does not cover.

Both obligations are attribution obligations. If you redistribute this pool,
carry them with it.
"""

from __future__ import annotations

from pathlib import Path

from . import mask  # noqa: F401

__all__ = ["load_pool", "declared_gates", "DECLARED", "mask", "MODEL_DIR"]

# frozenaudit/instances/abcb1/ -> repo root -> model/
MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "model"

DECLARED = {
    "endpoint": "ABCB1/P-glycoprotein efflux risk for a standardized "
                "free/released small-molecule structure",
    "model_status": {
        "v0.4_selective_uncertainty":
            "development_selective_risk_screener_not_future_source_validated",
        "v0.1_primary_screener":
            "development_research_screener_gate_failed",
        "v0.3_assay_aware_candidate":
            "single_source_internal_development_candidate_not_externally_validated",
    },
    "blind_gate_verdict": {
        "validated_binary_predictor_claim_status": "fail",
        "source": "abcb1_regulatory_v4_blind_summary_v1.json",
        "meaning": "the strict blind-pool gate on the binary substrate call did "
                   "not pass; this instance's bundles must not be presented as "
                   "validated binary substrate predictors",
    },
    "not_covered": "intact-ADC transport, intracellular unbound exposure, "
                   "resistance, efficacy, clinical DDI, or universal "
                   "assay-independent substrate truth",
    "better_free_alternative": {
        "model": "AstraZeneca GNN-MTL MDCK-ER checkpoint",
        "licence": "Apache-2.0",
        "why": "on the paper's external set it reaches 0.743 balanced accuracy "
               "against this instance's 0.582; if you want an efflux "
               "prediction rather than an audit, use it instead",
    },
    "reference_pool_licence": {
        "structures": "CC BY-SA 3.0 (ChEMBL; share-alike)",
        "labels": "CC BY 4.0 (Daood et al. 2025, "
                  "DOI 10.1021/acs.molpharmaceut.5c01065)",
        "evidence_tier": "B_author_curated_no_row_primary_assay",
    },
    "transfer_evidence": {
        "across_models": "five models, one endpoint (see paper Table 4)",
        "across_endpoints": "none; not attempted",
    },
}


def load_pool() -> list[dict]:
    """Build the 477-member joint reference pool from the shipped files.

    Reads the 443-row training audit CSV and the 63-row direct panel out of the
    v0.3 bundle, then de-duplicates on connectivity via
    :func:`frozenaudit.core.domain.build_pool`.

    Raises
    ------
    FileNotFoundError
        If ``model/`` is not next to the package -- i.e. the package was
        installed without its data files. The pool cannot be reconstructed
        from anything else, so this raises rather than returning a partial set.
    AssertionError
        If either source file no longer holds the row count the protocol was
        frozen with. A pool of a different size is a different protocol.
    """
    import joblib
    import pandas as pd

    from ...core import domain

    audit_csv = MODEL_DIR / "abcb1_screener_v0_1_training_audit.csv"
    direct_bundle = MODEL_DIR / "ABCB1_assay_aware_candidate_v0_3_dev.joblib"
    for path in (audit_csv, direct_bundle):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; the ABCB1 reference pool cannot be "
                "reconstructed without the shipped model/ directory"
            )

    audit = pd.read_csv(audit_csv, low_memory=False)
    assert len(audit) == 443, (
        f"primary reference must hold 443 rows, got {len(audit)}; "
        "this is a different protocol than the frozen one"
    )
    primary = [{"reference_id": r["interface_id"],
                "reference_name": r.get("preferred_name", ""),
                "standardized_smiles": r["derived_isomeric_smiles"],
                "standardized_inchikey": r["derived_standardized_inchikey"]}
               for _, r in audit.iterrows()]

    rows = joblib.load(direct_bundle)["training_rows"]
    assert len(rows) == 63, (
        f"direct panel must hold 63 rows, got {len(rows)}; "
        "this is a different protocol than the frozen one"
    )
    direct = [{"reference_id": r["panel_entity_id"],
               "reference_name": r.get("entity_name", ""),
               "standardized_smiles": r["standardized_isomeric_smiles"],
               "standardized_inchikey": r["standardized_inchikey"]}
              for r in rows]

    return domain.build_pool(primary + direct)


def declared_gates() -> list:
    """The out-of-distribution gates as declared in the paper, before scoring.

    Two of these three came back ``not_evaluable`` in the paper because the
    nearest stratum held 7 molecules. They are reproduced verbatim, including
    the ones that could not be evaluated, because a gate set edited down to
    the ones that resolved is no longer a pre-declared gate set.
    """
    from ...core import gates

    return [
        gates.Gate(
            name="ood_direction",
            declared="in the OOD stratum, the 95% lower bound of the "
                     "rank-correlation between score and measured efflux "
                     "ratio must exceed zero",
            min_n=20,
            rule=gates.ci_lower_above(0.0),
            requires=("ci95_low",),
        ),
        gates.Gate(
            name="transition_direction",
            declared="in the transition stratum, the 95% lower bound of the "
                     "same rank-correlation must exceed zero",
            min_n=20,
            rule=gates.ci_lower_above(0.0),
            requires=("ci95_low",),
        ),
        gates.Gate(
            name="extreme_ood_direction",
            declared="in the extreme-OOD stratum, the same lower bound must "
                     "exceed zero",
            min_n=20,
            rule=gates.ci_lower_above(0.0),
            requires=("ci95_low",),
        ),
    ]
