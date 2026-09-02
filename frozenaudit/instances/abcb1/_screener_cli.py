#!/usr/bin/env python3
"""Three-output ABCB1 efflux-risk research screener.

The operational outputs are high risk, low risk, and abstain. Abstention is a
decision policy, not a fitted biological class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, DataStructs


HERE = Path(__file__).resolve().parent
# Repackaged for standalone use: model files sit in ../model rather than in the
# original project tree. sys.path is extended so the two sibling helper modules
# resolve when this file is imported rather than run as a script.
# frozenaudit/instances/abcb1/ -> repo root, where model/ sits.
PROJECT = HERE.parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
_MODEL_DIR = PROJECT / "model"
SCRIPT_DIR = PROJECT / "数据" / "scripts"
V0_3_DIR = _MODEL_DIR
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V0_3_DIR))

from abcb1_screener_core_v0_1 import (  # noqa: E402
    FP_GENERATOR,
    descriptor_array,
    morgan_array,
    predict_smiles,
)
from predict_abcb1_assay_aware import predict_one as predict_direct_one  # noqa: E402


DEFAULT_SELECTIVE = _MODEL_DIR / "ABCB1_selective_uncertainty_v0_4_dev.joblib"
DEFAULT_PRIMARY = _MODEL_DIR / "ABCB1_screener_v0_1.joblib"
DEFAULT_DIRECT = V0_3_DIR / "ABCB1_assay_aware_candidate_v0_3_dev.joblib"

DECISION_ZH = {
    "high_ABCB1_efflux_risk": "高ABCB1外排风险",
    "low_ABCB1_efflux_risk": "低ABCB1外排风险",
    "abstain": "拒判",
}

REASON_ZH = {
    "unresolved_structure": "结构无法解析或标准化失败",
    "reference_identity_not_prospective": "与开发参考集完全同一，返回参考证据而非前瞻预测",
    "probability_gray_zone": "主模型概率位于高低风险阈值之间",
    "bootstrap_direction_unstable": "骨架bootstrap模型的方向区间跨越0.5",
    "outside_both_reference_panels": "同时低于两套参考面板的结构支持边界",
    "outside_physicochemical_envelope": "超出冻结理化性质包络",
    "supported_model_direction_disagreement": "在直接面板有结构支持时，两模型方向相反",
    "primary_local_neighborhood_strongly_opposite": "主参考集局部近邻方向与模型强烈相反",
    "direct_local_neighborhood_strongly_opposite": "直接实验参考集局部近邻方向与模型强烈相反",
    "stereo_blind_fingerprint_collision": "非手性指纹与不同立体身份完全相同",
    "object_scope_incompatible": "对象不是明确的游离载荷或释放小分子",
    "requested_context_incompatible": "请求端点超出ABCB1外排风险筛选范围",
}

ALLOWED_OBJECT_TYPES = {
    "released_or_free_small_molecule",
    "free_payload",
    "released_small_molecule",
    "released_linker_payload",
    "release_species",
    "small_molecule",
}
ALLOWED_REQUESTED_CONTEXTS = {
    "structure_only_screening",
    "ABCB1_transporter_assay_risk",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_packages(selective_path: Path, primary_path: Path, direct_path: Path) -> tuple[dict, dict, dict]:
    selective = joblib.load(selective_path)
    expected = selective["dependency_hashes"]
    observed_primary = sha256_file(primary_path)
    observed_direct = sha256_file(direct_path)
    expected_primary = expected["模型/ABCB1_screener_v0_1/ABCB1_screener_v0_1.joblib"]
    expected_direct = expected["模型/ABCB1_assay_aware_candidate_v0_3_dev/ABCB1_assay_aware_candidate_v0_3_dev.joblib"]
    if observed_primary != expected_primary:
        raise RuntimeError("frozen v0.1 hash mismatch")
    if observed_direct != expected_direct:
        raise RuntimeError("frozen v0.3 hash mismatch")
    return selective, joblib.load(primary_path), joblib.load(direct_path)


def calibrate_ensemble(raw_probability: np.ndarray, primary_package: dict) -> np.ndarray:
    calibrator = primary_package.get("calibrator")
    if calibrator is None:
        return raw_probability
    clipped = np.clip(raw_probability, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def bootstrap_probabilities(mol: Chem.Mol, selective: dict, primary: dict) -> np.ndarray:
    fp = morgan_array(mol).reshape(1, -1)
    descriptor = descriptor_array(mol).reshape(1, -1)
    raw: list[float] = []
    for item in selective["bootstrap_models"]:
        scaled = item["descriptor_scaler"].transform(descriptor)
        x = np.hstack([fp, scaled])
        raw.append(float(item["estimator"].predict_proba(x)[0, 1]))
    return calibrate_ensemble(np.asarray(raw, dtype=float), primary)


def local_neighborhood(
    query_fp,
    fingerprints: list,
    labels: list[int],
    reference_rows: list[dict],
    k: int,
) -> dict:
    similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, fingerprints), dtype=float)
    order = np.argsort(-similarities)[: max(1, min(k, len(similarities)))]
    weights = np.maximum(similarities[order], 1e-6) ** 2
    selected_labels = np.asarray([int(labels[int(index)]) for index in order], dtype=float)
    weighted_probability = float(np.sum(weights * selected_labels) / np.sum(weights))
    references: list[dict] = []
    for rank, index in enumerate(order, start=1):
        row = reference_rows[int(index)]
        references.append(
            {
                "rank": rank,
                "name": row.get("preferred_name", row.get("entity_name", "")),
                "label": int(labels[int(index)]),
                "tanimoto": round(float(similarities[int(index)]), 6),
                "standardized_inchikey": row.get("standardized_inchikey", ""),
                "evidence_tier": row.get("evidence_tier", ""),
                "source_url": row.get("source_url", ""),
            }
        )
    return {
        "max_similarity": float(similarities[order[0]]),
        "weighted_positive_fraction": weighted_probability,
        "conflict_index": float(4.0 * weighted_probability * (1.0 - weighted_probability)),
        "references": references,
    }


def physicochemical_support(mol: Chem.Mol, selective: dict) -> dict:
    values = descriptor_array(mol)
    center = np.asarray(selective["physchem_center"], dtype=float)
    scale = np.asarray(selective["physchem_scale"], dtype=float)
    robust_z = np.abs((values - center) / scale)
    score = float(np.max(robust_z))
    threshold = float(selective["physchem_score_threshold_q99"])
    return {
        "descriptor_values": {
            name: round(float(value), 6)
            for name, value in zip(selective["physchem_descriptor_names"], values)
        },
        "max_abs_robust_z": score,
        "threshold_q99": threshold,
        "within_envelope": score <= threshold,
    }


def direct_reference_rows(direct: dict) -> list[dict]:
    return [
        {
            "entity_name": row.get("entity_name", ""),
            "standardized_inchikey": row.get("standardized_inchikey", ""),
            "evidence_tier": "A1_matched_parental_direct_panel",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11207571/",
        }
        for row in direct["training_rows"]
    ]


def predict_one(
    smiles: str,
    selective: dict,
    primary: dict,
    direct: dict,
    chemical_object_type: str = "released_or_free_small_molecule",
    requested_context: str = "structure_only_screening",
) -> dict:
    primary_result = predict_smiles(smiles, primary)
    base = {
        "input_smiles": smiles or "",
        "standardization_status": primary_result.get("standardization_status", "failed"),
        "standardized_isomeric_smiles": primary_result.get("standardized_isomeric_smiles", ""),
        "standardized_inchikey": primary_result.get("standardized_inchikey", ""),
        "chemical_object_type": chemical_object_type,
        "requested_context": requested_context,
        "context_compatibility": "unresolved_structure",
        "risk_call": "abstain",
        "decision_zh": DECISION_ZH["abstain"],
        "accepted_prediction": "no",
        "primary_calibrated_probability": primary_result.get("calibrated_probability", ""),
        "primary_low_threshold": selective["primary_probability_low_threshold"],
        "primary_high_threshold": selective["primary_probability_high_threshold"],
        "bootstrap_probability_p10": "",
        "bootstrap_probability_median": "",
        "bootstrap_probability_p90": "",
        "bootstrap_interval_width": "",
        "direct_panel_probability": "",
        "primary_max_tanimoto": "",
        "direct_max_tanimoto": "",
        "primary_local_positive_fraction": "",
        "direct_local_positive_fraction": "",
        "primary_local_conflict_index": "",
        "direct_local_conflict_index": "",
        "physchem_max_abs_robust_z": "",
        "physchem_threshold_q99": selective["physchem_score_threshold_q99"],
        "applicability_status": "unresolved_structure",
        "uncertainty_status": "unresolved_structure",
        "exact_reference_identity": "no",
        "stereo_blind_collision_warning": "no",
        "abstention_reasons": ["unresolved_structure"],
        "abstention_reasons_zh": [REASON_ZH["unresolved_structure"]],
        "nearest_primary_references": [],
        "nearest_direct_references": [],
        "model_version": selective["model_version"],
        "validation_status": selective["status"],
        "claim_boundary": selective["claim_boundary"],
    }
    if primary_result.get("standardization_status") not in {"ok", "matched_frozen_training_structure"}:
        return base

    standardized_smiles = primary_result["standardized_isomeric_smiles"]
    mol = Chem.MolFromSmiles(standardized_smiles)
    if mol is None:
        return base
    query_fp = FP_GENERATOR.GetFingerprint(mol)
    direct_result = predict_direct_one(smiles, direct)
    primary_probability = float(primary_result["calibrated_probability"])
    direct_probability = float(direct_result["candidate_probability"])
    ensemble = bootstrap_probabilities(mol, selective, primary)
    p10, median, p90 = [float(value) for value in np.quantile(ensemble, selective["bootstrap_probability_quantiles"])]

    primary_local = local_neighborhood(
        query_fp,
        primary["training_fingerprints"],
        selective["primary_training_labels"],
        selective["primary_reference_rows"],
        int(selective["local_neighbor_k"]),
    )
    direct_rows = direct_reference_rows(direct)
    direct_local = local_neighborhood(
        query_fp,
        direct["training_fingerprints"],
        selective["direct_training_labels"],
        direct_rows,
        int(selective["local_neighbor_k"]),
    )
    physchem = physicochemical_support(mol, selective)

    query_key = primary_result["standardized_inchikey"]
    primary_identity = query_key in set(primary["training_structure_keys"])
    direct_identity = query_key in {row["standardized_inchikey"] for row in direct["training_rows"]}
    exact_identity = primary_identity or direct_identity
    stereo_collision = (
        (primary_local["max_similarity"] >= 0.999999 and query_key != primary_local["references"][0]["standardized_inchikey"])
        or (direct_local["max_similarity"] >= 0.999999 and query_key != direct_local["references"][0]["standardized_inchikey"])
    )

    low_threshold = float(selective["primary_probability_low_threshold"])
    high_threshold = float(selective["primary_probability_high_threshold"])
    if primary_probability >= high_threshold:
        direction = 1
    elif primary_probability <= low_threshold:
        direction = 0
    else:
        direction = None

    primary_supported = primary_local["max_similarity"] >= float(selective["primary_similarity_q10"])
    direct_supported = direct_local["max_similarity"] >= float(selective["direct_similarity_q10"])
    reasons: list[str] = []
    object_compatible = chemical_object_type in ALLOWED_OBJECT_TYPES
    context_compatible = requested_context in ALLOWED_REQUESTED_CONTEXTS
    if not object_compatible:
        reasons.append("object_scope_incompatible")
    if not context_compatible:
        reasons.append("requested_context_incompatible")
    if exact_identity:
        reasons.append("reference_identity_not_prospective")
    if direction is None:
        reasons.append("probability_gray_zone")
    elif direction == 1 and p10 < 0.5:
        reasons.append("bootstrap_direction_unstable")
    elif direction == 0 and p90 >= 0.5:
        reasons.append("bootstrap_direction_unstable")
    if not primary_supported and not direct_supported:
        reasons.append("outside_both_reference_panels")
    if not physchem["within_envelope"]:
        reasons.append("outside_physicochemical_envelope")
    if direction is not None and direct_supported and int(direct_probability >= 0.5) != direction:
        reasons.append("supported_model_direction_disagreement")
    if direction == 1 and primary_supported and primary_local["weighted_positive_fraction"] <= float(selective["local_low_boundary"]):
        reasons.append("primary_local_neighborhood_strongly_opposite")
    if direction == 0 and primary_supported and primary_local["weighted_positive_fraction"] >= float(selective["local_high_boundary"]):
        reasons.append("primary_local_neighborhood_strongly_opposite")
    if direction == 1 and direct_supported and direct_local["weighted_positive_fraction"] <= float(selective["local_low_boundary"]):
        reasons.append("direct_local_neighborhood_strongly_opposite")
    if direction == 0 and direct_supported and direct_local["weighted_positive_fraction"] >= float(selective["local_high_boundary"]):
        reasons.append("direct_local_neighborhood_strongly_opposite")
    if stereo_collision:
        reasons.append("stereo_blind_fingerprint_collision")
    reasons = list(dict.fromkeys(reasons))

    if not physchem["within_envelope"]:
        applicability = "outside_physicochemical_envelope"
    elif not primary_supported and not direct_supported:
        applicability = "outside_both_reference_panels"
    elif primary_supported and direct_supported:
        applicability = "supported_by_both_reference_panels"
    elif primary_supported:
        applicability = "supported_by_primary_reference_panel_only"
    else:
        applicability = "supported_by_direct_reference_panel_only"
    uncertainty = "direction_stable" if not (
        (direction == 1 and p10 < 0.5) or (direction == 0 and p90 >= 0.5) or direction is None
    ) else "uncertain_or_gray_zone"

    if reasons:
        risk_call = "abstain"
    else:
        risk_call = "high_ABCB1_efflux_risk" if direction == 1 else "low_ABCB1_efflux_risk"

    base.update(
        {
            "risk_call": risk_call,
            "decision_zh": DECISION_ZH[risk_call],
            "accepted_prediction": "yes" if risk_call != "abstain" else "no",
            "bootstrap_probability_p10": round(p10, 8),
            "bootstrap_probability_median": round(median, 8),
            "bootstrap_probability_p90": round(p90, 8),
            "bootstrap_interval_width": round(p90 - p10, 8),
            "direct_panel_probability": round(direct_probability, 8),
            "primary_max_tanimoto": round(primary_local["max_similarity"], 6),
            "direct_max_tanimoto": round(direct_local["max_similarity"], 6),
            "primary_local_positive_fraction": round(primary_local["weighted_positive_fraction"], 8),
            "direct_local_positive_fraction": round(direct_local["weighted_positive_fraction"], 8),
            "primary_local_conflict_index": round(primary_local["conflict_index"], 8),
            "direct_local_conflict_index": round(direct_local["conflict_index"], 8),
            "physchem_max_abs_robust_z": round(physchem["max_abs_robust_z"], 8),
            "context_compatibility": "compatible" if object_compatible and context_compatible else "incompatible_abstain",
            "applicability_status": applicability,
            "uncertainty_status": uncertainty,
            "exact_reference_identity": "yes" if exact_identity else "no",
            "stereo_blind_collision_warning": "yes" if stereo_collision else "no",
            "abstention_reasons": reasons,
            "abstention_reasons_zh": [REASON_ZH[reason] for reason in reasons],
            "nearest_primary_references": primary_local["references"],
            "nearest_direct_references": direct_local["references"],
            "physicochemical_descriptors": physchem["descriptor_values"],
        }
    )
    return base


def flatten_batch_result(result: dict) -> dict:
    nested = [
        "abstention_reasons",
        "abstention_reasons_zh",
        "nearest_primary_references",
        "nearest_direct_references",
        "physicochemical_descriptors",
    ]
    nested_keys = set(nested)
    flat = {key: value for key, value in result.items() if key not in nested_keys}
    for key in nested:
        flat[f"{key}_json"] = json.dumps(result.get(key), ensure_ascii=False, separators=(",", ":"))
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-output ABCB1 efflux-risk research screener")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles")
    source.add_argument("--input-csv", type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chemical-object-type", default="released_or_free_small_molecule")
    parser.add_argument("--requested-context", default="structure_only_screening")
    parser.add_argument("--object-type-column", default="chemical_object_type")
    parser.add_argument("--context-column", default="requested_context")
    parser.add_argument("--selective-model", type=Path, default=DEFAULT_SELECTIVE)
    parser.add_argument("--primary-model", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--direct-model", type=Path, default=DEFAULT_DIRECT)
    args = parser.parse_args()

    selective, primary, direct = load_packages(args.selective_model, args.primary_model, args.direct_model)
    if args.smiles is not None:
        print(
            json.dumps(
                predict_one(
                    args.smiles,
                    selective,
                    primary,
                    direct,
                    chemical_object_type=args.chemical_object_type,
                    requested_context=args.requested_context,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required with --input-csv")
    rows = read_csv(args.input_csv)
    if rows and args.smiles_column not in rows[0]:
        parser.error(f"SMILES column not found: {args.smiles_column}")
    output_rows = []
    for row in rows:
        object_type = row.get(args.object_type_column, "") or args.chemical_object_type
        context = row.get(args.context_column, "") or args.requested_context
        result = predict_one(
            row.get(args.smiles_column, ""),
            selective,
            primary,
            direct,
            chemical_object_type=object_type,
            requested_context=context,
        )
        output_rows.append({**row, **flatten_batch_result(result)})
    write_csv(args.output, output_rows)
    print(json.dumps({"input_rows": len(rows), "output": str(args.output), "model_version": selective["model_version"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
