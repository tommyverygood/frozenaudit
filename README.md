# EffluxScope

A computational framework for evaluating ABCB1 efflux predictors under
chemical-space shift.

EffluxScope combines structure-based stratification, coverage-matched
comparisons and pre-declared decision criteria to evaluate released
predictors. Its endpoint-agnostic core ships with an ABCB1/P-glycoprotein
worked instance. Reports record whether the evaluation rules were fixed
before scoring, keeping exploratory analyses distinct from confirmatory ones.

For compatibility, the Python distribution and import package remain named
`frozenaudit`; existing imports, APIs and environment variables continue to work.

```
frozenaudit/
├── core/          the evaluation workflow. No endpoint, no panel, no weights.
└── instances/     worked instances. One ships: ABCB1/P-gp efflux risk.
```

That split is the design. `core` is endpoint-agnostic and depends only on
NumPy; it would run the same way on a solubility model, a clearance model, or
something that is not chemistry at all. Everything property-specific — the
reference panel, the model bundles, the declared criteria — lives in an
instance.

---

## The five steps

| step | module | what it enforces |
| --- | --- | --- |
| 1. record | `core.freeze` | hash the evaluation specification, so drift between "the rules I declared" and "the rules I ran" is detectable rather than arguable |
| 2. stratify | `core.domain` | distance to a reference set, computed from structure alone — no measured outcome is read, so strata can be assigned before any label is known |
| 3. contrast | `core.contrast` | four arms at explicitly reported coverage, including the model's **own** confidence as the baseline the selector must beat |
| 4. gates | `core.gates` | pre-declared criteria that can return `not_evaluable`, so "we could not tell" never gets reported as "it failed" |
| 5. report | `core.protocol` | emit all of it, and state whether the rules were fixed before scoring |

Step 5 is the one that is easy to skip and expensive to omit.
`Audit.report()` records the order its steps ran in, and if scoring happened
before the rules were fixed it downgrades the whole run to `exploratory`,
regardless of the gate results. A gate that passes without rules fixed in
advance is a hypothesis, and the report says so instead of leaving the reader
to work it out.

### Step 3 in particular

The comparison that decides whether an abstention layer earns its place is
also the easiest one to fake. Score a model on everything, score it again on
the rows your selector kept, report the difference: that number is almost
always positive and almost always meaningless, because any selector that drops
rows looks good.

`core.contrast` therefore always computes four arms — `full`,
`selected`, `selected_complement`, and `native_margin_matched`, the last being
the same *number* of rows chosen instead by the model's own distance from its
decision threshold. A selector earns its place only by beating that arm,
because that arm is free.

---

## What this is not

**Not a property predictor.** The bundles under `frozenaudit/instances/abcb1/model/` are objects of
evaluation in the accompanying paper, not products of it. All three declare
themselves development artefacts, and
`instances.abcb1.mask.load_bundles()` writes those declarations to stderr on
every call so they appear before any number does:

```
v0.4  development_selective_risk_screener_not_future_source_validated
v0.1  development_research_screener_gate_failed
v0.3  single_source_internal_development_candidate_not_externally_validated
```

On the paper's external set of 2,624 identity-clean molecules the bundled
screener's own binary calls reach balanced accuracy **0.582** (MCC 0.168,
ROC-AUC 0.625) — second worst of the five models compared. A freely available
endpoint-aligned alternative, the AstraZeneca GNN-MTL MDCK-ER checkpoint
(Apache-2.0), reaches **0.743** on the same set. On a separate strict
regulatory blind pool (n = 33, model fixed and hash-verified before scoring)
the pre-declared gate on its binary calls returned **fail**.

So: **do not use this as a substrate predictor.** If you want an efflux-risk
score, use a better one. What this package is for is the *abstention* and the
evaluation around it.

**Not validated across endpoints.** The transfer evidence below is across
five *models* on one *endpoint*. `core` is written to be endpoint-agnostic and
`tests/test_core_general.py` exercises it on synthetic non-chemical data — but
"this code hard-codes no endpoint" is a statement about the code, not evidence
that the framework's conclusions carry to another property. A second worked
instance is open work. Do not cite the architecture as cross-endpoint
validation.

---

## When the shipped mask helps, and when it hurts

At the 17.3% coverage the mask selects, against each model's **own**
threshold-distance margin at that same coverage (balanced accuracy, same
external set):

| base model | no selection | own margin | fixed mask | mask − own margin |
| --- | --- | --- | --- | --- |
| Deep-PK P-gp substrate | 0.578 | 0.529 | 0.628 | **+0.098** |
| this project's v0.4 score | 0.582 | 0.637 | 0.735 | **+0.098** |
| ADMET-AI P-gp inhibition | 0.544 | 0.516 | 0.561 | **+0.045** |
| AZ GNN-MTL NIH-MDCK-ER | 0.699 | 0.703 | 0.736 | **+0.033** |
| AZ GNN-MTL MDCK-ER | 0.743 | 0.902 | 0.794 | **−0.108** |

Read the last row first. On a well-calibrated, endpoint-aligned score the
model's own distance from its threshold is the better selector and the mask
makes things worse. The mask earns its place when the base score is poorly
calibrated or its endpoint does not match the question being asked.

Note also that the mask beats *no selection* on all five models, including the
one where it loses to the free baseline. That is exactly why the free baseline
has to be in the table.

---

## Install

```bash
git clone https://github.com/tommyverygood/EffluxScope
cd EffluxScope
pip install -e ".[abcb1,test]"     # or ".[chem]", or bare "." for core only
```

Extras, because the core is deliberately light:

| extra | pulls in | for |
| --- | --- | --- |
| *(none)* | numpy | `core.freeze`, `core.contrast`, `core.gates`, `core.protocol` |
| `chem` | rdkit, pandas | adds `core.domain` — structure-only strata |
| `abcb1` | + scikit-learn, scipy, joblib | the shipped instance and its bundles |

`requirements-lock.txt` records the exact versions the reported numbers were
produced under. Newer RDKit releases can change fingerprint details; if the
reconciliation tests fail after an upgrade, pin to the lock file.

---

## Quick start

### Evaluate your own model, your own endpoint

No molecules required — this path needs only scores, labels and a threshold.

```python
import numpy as np
from frozenaudit.core import Audit, gates

audit = Audit("my_endpoint_v1", gates=[
    gates.Gate("direction", "95% lower bound of Spearman rho above zero",
               min_n=30, rule=gates.ci_lower_above(0.0),
               requires=("ci95_low",)),
])

# 1. record the rules -- before examining outcomes
lock = audit.freeze({"threshold": 0.5, "metric": "balanced_accuracy"})
print(lock.short)          # paste this into your methods section

# 3. contrast -- four arms, coverage reported
out = audit.contrast(scores=scores, labels=labels, threshold=0.5,
                     selector_accepted=my_selector_accepted)
print(out["reading"])
# -> "balanced_accuracy: the model's own margin beats selector by +0.063
#     at 25.0% coverage -- the external selector is not warranted on this model"

# 4. gates, then 5. report
audit.run_gates({"direction": {"n": 512, "ci95_low": 0.11}})
report = audit.report()
print(report["overall"], report["caveats"])
```

Add `pool=` at construction and call `audit.stratify(smiles)` for step 2.

### Reproduce the paper's selector

```python
from frozenaudit.instances.abcb1 import load_pool, mask, DECLARED

decisions = mask.decide(["CC(=O)Nc1ccc(O)cc1", "CN1CCC[C@H]1c1cccnc1"])
tagged = mask.apply_to(my_predictions, decisions)
print(DECLARED["blind_gate_verdict"])   # the fail verdict travels with the code
```

### Runtime

The three bundles were pickled with the versions in `requirements-lock.txt`:

```
python 3.12.13   scikit-learn 1.9.0   numpy 2.5.1   scipy 1.18.0
rdkit 2026.03.4  joblib 1.5.3
```

scikit-learn below 1.9 unpickles them without complaint and then fails inside
`predict_proba`. `mask.load_bundles()` therefore refuses to load on such a
runtime and says so, rather than letting every molecule come back as an
abstention -- abstaining is a legitimate output of this framework, so a broken
install must not be able to imitate one. `mask.runtime_problem()` returns that
diagnosis as a string, or `None` when the runtime is supported.

Every shipped file is checked against `model/manifest.json` on load
(`mask.verify_shipped_files()`); a mismatch raises rather than warns.

### Command line

```bash
python examples/audit_your_model.py examples/demo_predictions.csv
```

Input is a CSV with `smiles`, `score` (higher = more likely high-efflux) and
`observed` (1/0).

---

## Bringing your own endpoint

There is no plugin registry and no base class. An instance is a module
exposing three things by convention:

```python
def load_pool() -> list[dict]:     ...   # the reference set
def declared_gates() -> list[Gate]: ...  # criteria, written before scoring
DECLARED: dict                           # what the instance must disclose
```

Three things worth getting right, learned from building the one that ships:

**The pool is the evaluated model's training set, not a convenient public set.**
The whole meaning of "out of distribution" depends on which distribution. If
you do not know what a released model was trained on, say so in your report
rather than substituting something available and calling the result an
applicability domain.

**Keep the `declared` strings verbatim.** `Audit.freeze()` hashes them, so an
edit after the fact is detectable — which is the point. Rewording a gate to
match what you found is the failure mode this workflow is designed to detect.

**Do not invent a selector.** Most instances should not contribute one. The
honest default is the evaluated model's own margin, and `core.contrast` computes
it for free as the baseline arm.

And one on licensing: the moment an instance ships a reference panel, the
repository is redistributing someone's data and the MIT licence on the code
stops being the whole story. Check what you may ship before you ship it, and
record the answer in `DECLARED`.

---

## Reproducing the reported analysis

`core.domain` and `core.contrast` are faithful reimplementations of the
archived analysis specification, not independent re-derivations, and the test
suite proves it. Point
`FROZENAUDIT_FROZEN_DIR` at a directory holding the paper's archived result files:

```bash
FROZENAUDIT_FROZEN_DIR=/path/to/frozen pytest -q
```

| test | checks | against |
| --- | --- | --- |
| `test_pool_dedup_size` | 443 + 63 members collapse to 477 | — |
| `test_d5_matches_frozen` | per-molecule D5 and stratum | `abcb1_ood_structural_assignments_v1.csv` |
| `test_thresholds_match_frozen` | Q50/Q80/Q95 cut points | `abcb1_ood_reference_loo_distances_v1.csv` |
| `test_contrast_reproduces_published_table` | all three arms × five comparators, plus the native-margin selection row by row | `abcb1_public_predictor_metrics_v1.csv`, `..._predictions_long_v1.csv` |

Agreement is to machine precision. If any of these fails, **this package is
wrong and the archived result file is right.**

Without those files the reconciliation tests skip; `tests/test_core_general.py`
needs none of them and always runs.

---

## The reference pool

477 members: the 443-compound primary reference concatenated with the
63-compound direct-model panel, de-duplicated on InChIKey connectivity block
(29 stereoisomer duplicates collapse). Build it with
`instances.abcb1.load_pool()` — the same call the reconciliation test makes,
so it cannot drift from what users get.

The 443 primary compounds are ChEMBL records with substrate labels curated by
the authors of Daood et al. 2025. `evidence_tier` is
`B_author_curated_no_row_primary_assay` for every row: the label came from
that paper's curation of the literature, not from a per-row primary assay.
Treat the pool as a weak-label reference, not as measured ground truth.

---

## Licence

Four licences apply to different parts. This is not boilerplate — the
share-alike term on the ChEMBL structures binds anything derived from them,
and it does **not** extend to the rest of the pool.

| what | licence |
| --- | --- |
| code (`frozenaudit/`, `examples/`, `tests/`) | MIT — see `LICENSE` |
| the 443 ChEMBL reference structures | CC BY-SA 3.0 — **share-alike** |
| the substrate labels' curation | CC BY 4.0, Daood et al. 2025 |
| the 63-compound direct MDR1 panel | CC BY 4.0, Sóskuti et al. 2024 |

The 477-member pool is 443 + 63 de-duplicated, so it mixes the two. That is
permitted: CC lists BY → BY-SA as an allowed adapter's licence, and BY 4.0
imposes no share-alike of its own. But an adapter's licence reaches only your
own contributions, so redistributing the merged pool **does not** convert the
63 Sóskuti rows into share-alike material, and describing the whole pool as
CC BY-SA would impose a term CC BY 4.0 does not carry. Attribute each portion
to its own source and keep the share-alike obligation scoped to the ChEMBL
rows and their derivatives.

Both non-code portions have been modified — standardised, recomputed and
binarised — as `THIRD_PARTY_NOTICES.md` records per source. Full attribution
and exact obligations are there.

---

## Citing

Please cite the accompanying paper. Once it has a DOI this section will carry
it; until then cite this repository and its commit.

The reference panel's sources must be cited independently of this package —
ChEMBL and Daood et al. 2025, both in `THIRD_PARTY_NOTICES.md`.
