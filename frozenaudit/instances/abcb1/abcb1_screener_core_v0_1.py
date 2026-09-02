#!/usr/bin/env python3
"""Shared chemistry and prediction helpers for the ABCB1 research screener v0.1."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


MORGAN_RADIUS = 2
MORGAN_BITS = 2048
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
    includeChirality=False,
)
TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
UNCHARGER = rdMolStandardize.Uncharger(canonicalOrder=True)
RDLogger.DisableLog("rdApp.warning")


def _formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def standardize_smiles(smiles: str) -> dict:
    """Apply the project v1 standardization sequence without mutating the input."""
    result = {
        "input_smiles": smiles or "",
        "standardization_status": "failed",
        "standardized_isomeric_smiles": "",
        "standardized_connectivity_smiles": "",
        "standardized_inchikey": "",
        "fragment_removed_flag": "",
        "tautomer_status": "not_run",
        "standardization_error": "",
    }
    if not smiles or not str(smiles).strip():
        result["standardization_error"] = "blank_smiles"
        return result
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            result["standardization_error"] = "rdkit_parse_failed"
            return result
        source_fragments = len(Chem.GetMolFrags(mol))
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol, skipStandardize=True)
        mol = UNCHARGER.uncharge(mol)
        tautomer_result = TAUTOMER_ENUMERATOR.Enumerate(mol)
        mol = TAUTOMER_ENUMERATOR.PickCanonical(tautomer_result)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        connectivity_mol = Chem.Mol(mol)
        Chem.RemoveStereochemistry(connectivity_mol)
        connectivity = Chem.MolToSmiles(connectivity_mol, canonical=True, isomericSmiles=False)
        result.update(
            {
                "standardization_status": "ok",
                "standardized_isomeric_smiles": iso,
                "standardized_connectivity_smiles": connectivity,
                "standardized_inchikey": Chem.MolToInchiKey(mol),
                "fragment_removed_flag": "yes" if source_fragments > 1 else "no",
                "tautomer_status": getattr(tautomer_result.status, "name", str(tautomer_result.status)),
            }
        )
        return result
    except Exception as exc:  # pragma: no cover - records the chemistry failure for audit.
        result["standardization_error"] = f"{type(exc).__name__}: {exc}"
        return result


def mol_from_standardized(record: dict) -> Chem.Mol:
    mol = Chem.MolFromSmiles(record.get("standardized_isomeric_smiles", ""))
    if mol is None:
        raise ValueError("standardized structure is not parseable")
    return mol


def morgan_array(mol: Chem.Mol) -> np.ndarray:
    arr = np.zeros((MORGAN_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(FP_GENERATOR.GetFingerprint(mol), arr)
    return arr


def descriptor_array(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            _formal_charge(mol),
            mol.GetNumHeavyAtoms(),
        ],
        dtype=np.float64,
    )


def max_training_similarity(mol: Chem.Mol, packed_fingerprints: list) -> float:
    if not packed_fingerprints:
        return 0.0
    fp = FP_GENERATOR.GetFingerprint(mol)
    return float(max(DataStructs.BulkTanimotoSimilarity(fp, packed_fingerprints)))


def _raw_probability(package: dict, mol: Chem.Mol) -> float:
    feature_type = package["feature_type"]
    if feature_type == "morgan":
        x = morgan_array(mol).reshape(1, -1)
    elif feature_type == "physchem":
        x = package["descriptor_scaler"].transform(descriptor_array(mol).reshape(1, -1))
    elif feature_type == "morgan_physchem":
        desc = package["descriptor_scaler"].transform(descriptor_array(mol).reshape(1, -1))
        x = np.hstack([morgan_array(mol).reshape(1, -1), desc])
    else:
        raise ValueError(f"unsupported feature type: {feature_type}")
    return float(package["estimator"].predict_proba(x)[0, 1])


def predict_smiles(smiles: str, package: dict) -> dict:
    frozen_map = package.get("training_structure_map", {})
    direct_mol = Chem.MolFromSmiles(str(smiles).strip()) if smiles and str(smiles).strip() else None
    direct_inchikey = Chem.MolToInchiKey(direct_mol) if direct_mol is not None else ""
    if direct_inchikey and direct_inchikey in frozen_map:
        frozen_smiles = frozen_map[direct_inchikey]
        frozen_mol = Chem.MolFromSmiles(frozen_smiles)
        connectivity_mol = Chem.Mol(frozen_mol)
        Chem.RemoveStereochemistry(connectivity_mol)
        standardized = {
            "input_smiles": smiles,
            "standardization_status": "matched_frozen_training_structure",
            "standardized_isomeric_smiles": frozen_smiles,
            "standardized_connectivity_smiles": Chem.MolToSmiles(connectivity_mol, canonical=True, isomericSmiles=False),
            "standardized_inchikey": direct_inchikey,
            "fragment_removed_flag": "no",
            "tautomer_status": "frozen_training_identity_match",
            "standardization_error": "",
        }
    else:
        standardized = standardize_smiles(smiles)
    base = {
        **standardized,
        "transporter_gene": "ABCB1",
        "raw_probability": "",
        "calibrated_probability": "",
        "binary_prediction": "",
        "binary_threshold": package.get("binary_threshold", 0.5),
        "max_training_morgan_tanimoto": "",
        "applicability_band": "unresolved_structure",
        "screening_call": "unresolved_structure",
        "claim_boundary": package.get("claim_boundary", "research_screening_only"),
    }
    if standardized["standardization_status"] not in {"ok", "matched_frozen_training_structure"}:
        return base
    mol = mol_from_standardized(standardized)
    raw_probability = _raw_probability(package, mol)
    calibrator = package.get("calibrator")
    if calibrator is None:
        calibrated = raw_probability
    else:
        clipped = float(np.clip(raw_probability, 1e-6, 1 - 1e-6))
        logit = math.log(clipped / (1.0 - clipped))
        calibrated = float(calibrator.predict_proba(np.asarray([[logit]]))[:, 1][0])
    similarity = max_training_similarity(mol, package.get("training_fingerprints", []))
    q10 = float(package["applicability_thresholds"]["self_nearest_q10"])
    q25 = float(package["applicability_thresholds"]["self_nearest_q25"])
    threshold = float(package.get("binary_threshold", 0.5))
    binary = int(calibrated >= threshold)
    if similarity < q10:
        band = "outside_empirical_support"
        call = "abstain_outside_domain"
    elif similarity < q25:
        band = "low_empirical_support"
        call = f"caution_{'substrate' if binary else 'non_substrate'}"
    else:
        band = "within_empirical_support"
        call = "substrate" if binary else "non_substrate"
    base.update(
        {
            "raw_probability": round(raw_probability, 8),
            "calibrated_probability": round(calibrated, 8),
            "binary_prediction": binary,
            "max_training_morgan_tanimoto": round(similarity, 6),
            "applicability_band": band,
            "screening_call": call,
        }
    )
    return base


def load_package(path: str | Path) -> dict:
    return joblib.load(Path(path))
