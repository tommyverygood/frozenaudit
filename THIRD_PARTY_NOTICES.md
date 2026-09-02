# Third-party notices

Three separate licences apply to different files in this repository. The
share-alike term in section 1 is binding on anything you redistribute.

## 1. Reference structures — ChEMBL (CC BY-SA 3.0)

`model/abcb1_screener_v0_1_training_audit.csv` and the reference rows and
fingerprints embedded in `model/ABCB1_screener_v0_1.joblib` and
`model/ABCB1_selective_uncertainty_v0_4_dev.joblib` are derived from ChEMBL
records.

* Source: ChEMBL, European Molecular Biology Laboratory --- European
  Bioinformatics Institute, <https://www.ebi.ac.uk/chembl/>
* Accessed: April 2025, by the curators of the source publication in section 2,
  via `chembl_webresource_client`. The upstream publication reports an access
  date rather than a numbered release, so no ChEMBL release number can be
  stated here without inventing one.
* Licence: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0),
  <https://creativecommons.org/licenses/by-sa/3.0/>

**Obligation.** CC BY-SA 3.0 permits redistribution and adaptation with
attribution, and requires that adaptations be released under the same licence.
If you redistribute the reference panel, the fingerprints derived from it, or
any adaptation of either, that portion must carry CC BY-SA 3.0 and attribute
ChEMBL. The MIT licence on this repository's code does not extend to it.

## 2. Substrate labels --- Daood et al. 2025 (CC BY 4.0)

The substrate / non-substrate label on each of the 443 reference compounds
comes from the manual curation reported in:

> Daood et al. Machine Learning Modeling for ABC Transporter Efflux and
> Inhibition: Data Curation, Model Development, and New Compound Interaction
> Predictions. *Molecular Pharmaceutics*, 2025.
> DOI: 10.1021/acs.molpharmaceut.5c01065

* Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
  <https://creativecommons.org/licenses/by/4.0/>, as recorded in PubMed Central
  (PMC12587445).

Every reference row carries `evidence_tier =
B_author_curated_no_row_primary_assay`: the label reflects that paper's reading
of the literature, not a primary assay measured for this work. Cite the paper
above whenever the pool's labels are used or reported.

## 3. Direct-evidence panel

`model/ABCB1_assay_aware_candidate_v0_3_dev.joblib` embeds a 63-compound panel
assembled from published same-system MDR1 measurements. Those measurements
remain the property of their original publishers and are cited in the
accompanying paper; the panel is redistributed here only as the model's frozen
internal state.

## 4. Compared models, not redistributed

The performance table in `README.md` reports numbers for models that are **not**
included in this repository: the AstraZeneca GNN-MTL MDCK-ER and NIH-MDCK-ER
checkpoints (Apache-2.0, obtain from their own release), Deep-PK, and ADMET-AI.
Only the measured numbers are reproduced here; obtain each model from its own
distribution under its own terms.
