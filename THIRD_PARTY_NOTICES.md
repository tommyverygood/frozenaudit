# Third-party notices

Three separate licences apply to different files in this repository. The
share-alike term in section 1 is binding on anything you redistribute.

## 1. Reference structures — ChEMBL (CC BY-SA 3.0)

`frozenaudit/instances/abcb1/model/abcb1_screener_v0_1_training_audit.csv` and the reference rows and
fingerprints embedded in `frozenaudit/instances/abcb1/model/ABCB1_screener_v0_1.joblib` and
`frozenaudit/instances/abcb1/model/ABCB1_selective_uncertainty_v0_4_dev.joblib` are derived from ChEMBL
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

## 3. Direct-evidence panel --- Sóskuti et al. 2024 (CC BY 4.0)

`frozenaudit/instances/abcb1/model/ABCB1_assay_aware_candidate_v0_3_dev.joblib`
embeds a 63-compound panel of same-system MDR1 measurements as its frozen
`training_rows`, and
`frozenaudit/instances/abcb1/model/ABCB1_selective_uncertainty_v0_4_dev.joblib`
carries the same panel's 63 binary labels as `direct_training_labels`. Both
therefore redistribute this source.

* Source: Sóskuti E, Szilvásy N, Temesszentandrási-Ambrus C, Urbán Z,
  Csíkvári O, Szabó Z, Kecskeméti G, Pusztai É, Gáborik Z. Applicability of
  MDR1 Overexpressing Abcb1KO-MDCKII Cell Lines for Investigating In Vitro
  Species Differences and Brain Penetration Prediction. *Pharmaceutics*
  2024;16(6):736. DOI 10.3390/pharmaceutics16060736. PMID 38931858;
  PMCID PMC11207571.
* Taken from: supplementary Table S1, "ERs across the six cell lines for 83
  compounds (1 µM, 120 min)",
  <https://www.mdpi.com/article/10.3390/pharmaceutics16060736/s1>
* Licence: CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>
* Copyright notice, as supplied by the licensor: "© 2024 by the authors.
  Licensee MDPI, Basel, Switzerland. This article is an open access article
  distributed under the terms and conditions of the Creative Commons
  Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/)."
* Warranty: the Licensed Material is offered as-is and as-available; see
  Sections 5 and 6 of the licence.

MDPI states that the CC BY licence covers supplementary material: "all
articles published in MDPI journals, including their data, graphics, and
supplementary material, can be linked by external sources, scanned by search
engines, and reused by text mining applications, websites, or blogs, free of
charge under the sole condition that the source and original publisher are
properly accredited" (<https://www.mdpi.com/openaccess>). The supplementary
PDF itself carries no licence notice, so that policy statement and the
article-level licence are what this attribution rests on.

**Modifications.** CC BY 4.0 Section 3(a)(1)(B) requires that modification be
indicated, and this panel is heavily modified. The values were parsed from the
supplementary PDF; structures were resolved and standardised; a net efflux
ratio was computed as the hMDR1 ratio over the matched Mock ratio; that ratio
was binarised at the ICH M12 cut-off of 2 into a screening-evidence label; and
63 of the 84 parsed entities were retained, 21 being dropped as QC-flagged or
interval-crossing. The stored field is `log2_net_efflux_ratio`. No
compound-level inhibitor rescue is reported in the source, so a ratio at or
above 2 is recorded as screening evidence, not as a confirmed substrate call.

CC BY 4.0 does not restrict uses that need no permission, and measured values
are facts; this notice attributes the source rather than asserting that every
number in it is separately protected.

## 4. Compared models, not redistributed

The performance table in `README.md` reports numbers for models that are **not**
included in this repository: the AstraZeneca GNN-MTL MDCK-ER and NIH-MDCK-ER
checkpoints (Apache-2.0, obtain from their own release), Deep-PK, and ADMET-AI.
Only the measured numbers are reproduced here; obtain each model from its own
distribution under its own terms.
