#!/usr/bin/env python3
"""CLI for the ABCB1 assay-aware single-source development candidate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, DataStructs


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SCRIPT_DIR = PROJECT / "数据/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from abcb1_screener_core_v0_1 import (  # noqa: E402
    FP_GENERATOR,
    descriptor_array,
    morgan_array,
    standardize_smiles,
)


def predict_one(smiles: str, package: dict) -> dict:
    standardized = standardize_smiles(smiles)
    result = {
        **standardized,
        "transporter_gene": "ABCB1",
        "assay_context": "human MDR1 overexpressing Abcb1KO-MDCKII versus matched Mock; 1 uM; 120 min",
        "candidate_probability": "",
        "binary_threshold": package["binary_threshold"],
        "candidate_binary_prediction": "",
        "max_direct_panel_tanimoto": "",
        "applicability_band": "unresolved_structure",
        "triage_call": "unresolved_structure",
        "nearest_direct_reference_name": "",
        "nearest_direct_reference_label": "",
        "nearest_direct_reference_net_efflux_ratio": "",
        "nearest_direct_reference_tanimoto": "",
        "nearest_direct_reference_inchikey": "",
        "exact_development_identity_match": "no",
        "model_version": package["model_version"],
        "validation_status": package["status"],
        "claim_boundary": package["claim_boundary"],
    }
    if standardized["standardization_status"] != "ok":
        return result
    mol = Chem.MolFromSmiles(standardized["standardized_isomeric_smiles"])
    fp_array = morgan_array(mol).reshape(1, -1)
    desc = descriptor_array(mol).reshape(1, -1)
    feature_type = package["feature_type"]
    if feature_type == "physchem":
        x = package["descriptor_scaler"].transform(desc)
    elif feature_type == "morgan":
        x = fp_array
    elif feature_type == "morgan_plus_physchem":
        x = np.hstack([fp_array, package["descriptor_scaler"].transform(desc)])
    else:
        raise ValueError(f"unsupported feature type: {feature_type}")
    probability = float(package["estimator"].predict_proba(x)[0, 1])
    binary = int(probability >= float(package["binary_threshold"]))
    query_fp = FP_GENERATOR.GetFingerprint(mol)
    similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, package["training_fingerprints"]), dtype=float)
    nearest_index = int(np.argmax(similarities))
    similarity = float(similarities[nearest_index])
    reference = package["training_rows"][nearest_index]
    q10 = float(package["applicability_thresholds"]["self_nearest_q10"])
    q25 = float(package["applicability_thresholds"]["self_nearest_q25"])
    exact = standardized["standardized_inchikey"] == reference["standardized_inchikey"]
    if exact:
        band = "development_training_identity"
        call = "training_identity_reference_only_not_validation"
    elif similarity < q10:
        band = "outside_direct_panel_empirical_support"
        call = "abstain_outside_direct_panel_domain"
    elif similarity < q25:
        band = "low_direct_panel_empirical_support"
        call = "manual_review_low_direct_panel_support"
    else:
        band = "within_direct_panel_empirical_support"
        call = "candidate_positive_like" if binary else "candidate_negative_like"
    result.update(
        {
            "candidate_probability": round(probability, 8),
            "candidate_binary_prediction": binary,
            "max_direct_panel_tanimoto": round(similarity, 6),
            "applicability_band": band,
            "triage_call": call,
            "nearest_direct_reference_name": reference["entity_name"],
            "nearest_direct_reference_label": reference["target_label"],
            "nearest_direct_reference_net_efflux_ratio": round(float(2 ** reference["log2_net_efflux_ratio"]), 6),
            "nearest_direct_reference_tanimoto": round(similarity, 6),
            "nearest_direct_reference_inchikey": reference["standardized_inchikey"],
            "exact_development_identity_match": "yes" if exact else "no",
        }
    )
    return result


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ABCB1 matched-parental assay-aware research screener candidate")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles")
    source.add_argument("--input-csv", type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path, default=HERE / "ABCB1_assay_aware_candidate_v0_3_dev.joblib")
    args = parser.parse_args()
    package = joblib.load(args.model)
    if args.smiles is not None:
        print(json.dumps(predict_one(args.smiles, package), ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error("--output is required with --input-csv")
    rows = read_csv(args.input_csv)
    if rows and args.smiles_column not in rows[0]:
        parser.error(f"SMILES column not found: {args.smiles_column}")
    write_csv(args.output, [{**row, **predict_one(row.get(args.smiles_column, ""), package)} for row in rows])
    print(json.dumps({"input_rows": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
