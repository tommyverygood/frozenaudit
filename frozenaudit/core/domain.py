"""Distance to the frozen reference pool, and structure-only strata.

Every quantity here is computed from structure alone. Nothing in this module
reads a measured outcome, so the strata it returns can be assigned to a query
set before any label is known.

This is a faithful reimplementation of the frozen protocol used in the
accompanying paper, not an independent re-derivation. ``tests/`` reconciles it
against the frozen per-molecule file; if that test fails, this module is wrong
and the paper's file is right.

Protocol, as locked before any structure-only assignment was generated:

* fingerprint   -- Morgan, radius 2, 2048 bits, ``includeChirality=False``,
                   taken from the standardized *isomeric* SMILES
* connectivity  -- the first hyphen-delimited block of the standardized
                   InChIKey; this is the unit of de-duplication, so
                   stereoisomers collapse to one pool member
* joint pool    -- the 443-member primary reference concatenated with the
                   63-member direct-model panel, then de-duplicated on
                   connectivity keeping the first occurrence
* S1, S5        -- Tanimoto to the nearest, and the mean over the 5 nearest,
                   pool members
* D1, D5        -- 1 - S1 and 1 - S5
* thresholds    -- Q50 / Q80 / Q95 of the pool's own leave-one-*connectivity*-out
                   D5 distribution. Leaving out the connectivity block rather
                   than the row matters: a stereoisomer of the query would
                   otherwise sit at distance zero from it.
* strata        -- ID_like <= Q50 < transition <= Q80 < OOD <= Q95 < extreme_OOD
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_BITS = 2048
FP_CHIRALITY = False
TOP_K = 5
LOHI_SIMILARITY_CUTOFF = 0.40
STRATUM_NAMES = ("ID_like", "transition", "OOD", "extreme_OOD")

_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_BITS, includeChirality=FP_CHIRALITY
)


def fingerprint(smiles: str):
    """Morgan fingerprint of one SMILES, or None if RDKit cannot parse it.

    A None return is not an error: the frozen decision rule abstains on any
    structure that fails to parse, so callers must treat None as abstain and
    never as "very distant".
    """
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _GEN.GetFingerprint(mol)


def connectivity_block(inchikey, smiles: str) -> str:
    """De-duplication key: the InChIKey's first block, else a canonical SMILES.

    The first block encodes the molecular skeleton without stereochemistry, so
    stereoisomers share a key and collapse to one pool member.
    """
    value = "" if inchikey is None or str(inchikey).strip().lower() in ("", "nan") \
        else str(inchikey).strip()
    if value:
        return value.split("-", 1)[0]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot build a connectivity key for {smiles!r}")
    return "SMILES::" + Chem.MolToSmiles(mol, isomericSmiles=False)


def top_similarity(query_fp, reference_fps, k: int = TOP_K):
    """(S1, S5, top-k list) of Tanimoto similarities against a pool."""
    from rdkit import DataStructs

    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, list(reference_fps)),
                      dtype=float)
    if sims.size < k:
        raise ValueError(f"reference pool has fewer than {k} structures")
    top = np.sort(sims)[::-1][:k]
    return float(top[0]), float(np.mean(top)), [float(x) for x in top]


def build_pool(members: Sequence[dict]) -> list[dict]:
    """De-duplicate pool members on connectivity, keeping the first occurrence.

    Each member needs ``standardized_smiles`` and ``standardized_inchikey``.
    Returns the surviving members with ``connectivity_block`` and ``fp`` added.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for row in members:
        block = connectivity_block(row.get("standardized_inchikey"),
                                   row["standardized_smiles"])
        if block in seen:
            continue
        fp = fingerprint(row["standardized_smiles"])
        if fp is None:
            raise ValueError(f"unparseable pool structure: {row['standardized_smiles']!r}")
        seen.add(block)
        out.append({**row, "connectivity_block": block, "fp": fp})
    return out


def leave_one_connectivity_out_d5(pool: Sequence[dict], k: int = TOP_K) -> np.ndarray:
    """D5 of each pool member against the pool with its whole connectivity removed."""
    fps = [r["fp"] for r in pool]
    blocks = [r["connectivity_block"] for r in pool]
    out = np.empty(len(pool), dtype=float)
    for i, q in enumerate(fps):
        keep = [fps[j] for j, blk in enumerate(blocks) if blk != blocks[i]]
        if len(keep) < k:
            raise ValueError("too few leave-one-connectivity-out references")
        _, s5, _ = top_similarity(q, keep, k)
        out[i] = 1.0 - s5
    return out


def d5_thresholds(loo_d5: np.ndarray) -> dict:
    """Q50 / Q80 / Q95 cut points of a leave-one-connectivity-out D5 array."""
    if loo_d5.size < 20:
        raise ValueError("too few leave-one-out rows to set thresholds")
    return {"q50": float(np.quantile(loo_d5, 0.50)),
            "q80": float(np.quantile(loo_d5, 0.80)),
            "q95": float(np.quantile(loo_d5, 0.95)),
            "min": float(np.min(loo_d5)), "max": float(np.max(loo_d5))}


def distance_stratum(d5: float, thresholds: dict) -> str:
    """Structure-only stratum of one query D5."""
    if d5 <= thresholds["q50"]:
        return STRATUM_NAMES[0]
    if d5 <= thresholds["q80"]:
        return STRATUM_NAMES[1]
    if d5 <= thresholds["q95"]:
        return STRATUM_NAMES[2]
    return STRATUM_NAMES[3]


def score_queries(query_smiles: Iterable[str], pool: Sequence[dict],
                  thresholds: dict, k: int = TOP_K) -> list[dict]:
    """S1/S5/D1/D5 and stratum for each query.

    An unparseable query yields ``parsed=False`` with every distance None and
    ``stratum=None`` -- the caller must abstain on it rather than substitute a
    value.
    """
    fps_pool = [r["fp"] for r in pool]
    out: list[dict] = []
    for smiles in query_smiles:
        fp = fingerprint(smiles)
        if fp is None:
            out.append({"smiles": smiles, "parsed": False, "s1": None, "s5": None,
                        "d1": None, "d5": None, "stratum": None})
            continue
        s1, s5, _ = top_similarity(fp, fps_pool, k)
        d5 = 1.0 - s5
        out.append({"smiles": smiles, "parsed": True, "s1": s1, "s5": s5,
                    "d1": 1.0 - s1, "d5": d5,
                    "stratum": distance_stratum(d5, thresholds),
                    "strict_lohi_ood": s1 <= LOHI_SIMILARITY_CUTOFF})
    return out
