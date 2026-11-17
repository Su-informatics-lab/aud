# Provenance

Which script produces which result in the paper.

Everything here runs against protected health information on a university cluster. The
data are not in this repository and cannot be released, so the paths in the code point
at files that exist only there.

## 1. Inputs

The analysis starts from these, all outside the repository:

| input | what it is |
|---|---|
| `chart/ratings/*.json`, `resources/patient_roster.csv` | the hepatologists' classifications of each chart |
| `output/panel.csv` | one row per patient per engine: the six-grade classification and its class, the rule findings, the quotation and the note type or section cited, the engine's stated confidence and reasoning, and the decoding temperature and top-p (temperature 0 throughout) |
| `output/skill_variations.csv` | one row per patient per skill variation for the primary engine: the classification, and the temperature and top-p it was produced at (13 charts under the paraphrases were repeated at temperature 0.1, top-p 0.95) |
| the registry tables (`registry_frame.D`) | diagnoses, visits, drug records and deaths of the 14,406 registry patients |
| the note text | read only by the quotation audit and the two note comparators |
| `resources/institution_report.xlsx`, `output/fairness/patient_institution.csv` | the hospital system that wrote each note, and each patient's notes |
| `output/features_structured.parquet`, `output/sdoh/`, `output/geo/`, `resources/sdi_zcta.csv` | structured-record features and neighbourhood deprivation, for the comparators and the patient groups |

How the model classifications were produced (the engine serving, the harness, the note
extraction) is not released. Everything downstream treats a classification as data.

## 2. Rebuild

```sh
bash rebuild.sh
```

One script, three steps, in dependency order:

1. ground truth, agreement, accuracy, patient groups and comparators (`downstream/`);
2. the registry estimate, treatment and the sensitivity analyses (`registry/`);
3. the figure panels (`figures/`).

The quotation audit and the two note comparators are listed in the script and run
separately: the first reads the note text, the others need a batch job.

## 3. The ground truth

```
downstream/npj/ref_standard.py         the ground truth, derived in one place
downstream/npj/registry_frame_ids.py   which sampled patients are in the registry frame
```

Hepatologist A read every chart. A chart went to a second reading for either of two
reasons, and where the two readings fell in different classes a third hepatologist
decided. `ref_standard.build()` applies that rule, and every other script calls it.

| | charts |
|---|---|
| study cohort | 485 |
| ground truth AUD+ | 250 |
| ground truth AUD− | 173 |
| ground truth indeterminate | 62 |
| determinate, scored for sensitivity and specificity | 423 |

## 4. The registry frame and the registry estimate

```
downstream/registry_frame.py           t0, the strata, the code sets, the drug definition, follow-up
registry/source_table.py               one row per registry patient, with hospital type and whole-record diagnosis status
registry/registry_estimate.py          the nine-cell estimate and everything counted over it
registry/publish_tables.py             writes the results into the tables the figure producers read
registry/predictive_values.py          registry-weighted predictive values and the rates behind Fig. 1f
registry/t0_summary.py                 the range of t0 and the strata at t0
registry/medication_before_t0.py       medication recorded before t0, by stratum
```

`registry_frame.py` states the AUD diagnosis, the alcohol-related diagnosis code set
(Supplementary Table 15) and the AUD-specific drugs once; every registry-scale number
goes through it. `registry_estimate.py` weights each chart with a determinate ground
truth to the registry patients of its cell and writes
`output/registry/master_numbers_2026-10-26.json`, which holds the registry estimate
(6,223 patients with AUD), its split by stratum and hospital type, the registry-weighted
accuracy and the treatment shares.

Checked on the cluster from a clean directory: `rebuild.sh` runs to the end, the outputs of step 2 equal the ones
the paper was written from, and so do the result tables of step 1. The figure panels of step 3 were drawn without error
and were not compared image by image.

## 5. Display items

Panels are drawn by the scripts below and placed on the page by hand in a slide deck.
Flow diagrams and schematics are drawn by hand from the counts the named script prints.

### Main figures

| item | script |
|---|---|
| Fig. 1a, b | drawn by hand; counts from `downstream/npj/f00_cohort_flow_and_table1.py` |
| Fig. 1c | `figures/panel_b.py`, from `downstream/npj/f01_accuracy_against_reference.py` |
| Fig. 1d | `figures/panel_d.py`, from `registry/registry_estimate.py` and `registry/publish_tables.py` |
| Fig. 1e | `figures/panel_e.py`, from `downstream/npj/panels_de.py` |
| Fig. 1f | `figures/panel_f.py`, from `registry/predictive_values.py` (registry-weighted rates) |
| Fig. 1g | `figures/panel_g_v15.py`, from `downstream/npj/build_supp_table6.py` |
| Fig. 2a | drawn by hand |
| Fig. 2b, c | `figures/fig2nh_ab.py`; the MedDx-based estimate from `registry/meddx_estimate_and_band.py`, the sensitivities from `registry/sensitivity_rows.py` |
| Fig. 2d, e | `figures/fig2nh_de_v16.py`, from `registry/registry_estimate.py`, `registry/diagnosis_accuracy_and_curves.py` and `registry/drug_extraction_check.py` |
| Fig. 3a | `figures/panel_subgroup_disparity.py`, from `downstream/npj/f04_subgroup_disparity.py` |
| Fig. 3b | `figures/panel_fig3_d.py` |
| Fig. 3c | `figures/panel_site.py`, from `downstream/npj/f04_subgroup_rates.py` |
| Fig. 3d | `figures/panel_roc_space.py` |
| Fig. 3e | `figures/panel_transfer.py`, from `downstream/npj/site_transfer.py` |
| Fig. 4a | drawn by hand |
| Fig. 4b | `figures/panel_model_leaderboard_v16.py`, from `downstream/npj/f04_figure4.py` |
| Fig. 4c | `figures/fig4_v15.py`, from `figures/acc_readers.py` |
| Fig. 4d | `figures/panel_model_matrix_v15.py`, from `downstream/build_npj_materials.py` |
| Fig. 4e | `figures/panel_stability_v15.py`, from `figures/stability.py` |
| Fig. 4f | `figures/fig4_v15.py`, from `figures/panel_audit_v8.py`, `downstream/run_npj_provenance.py` and `downstream/grounding_finish.py` |
| Fig. 4g | `figures/fig4_v15.py`, from `figures/panel_trigger_section.py` (note types or sections cited, by ground-truth class) |

### Extended Data

| item | script |
|---|---|
| Extended Data Fig. 1 | drawn by hand; counts from `downstream/npj/f00_cohort_flow_and_table1.py` |
| Extended Data Fig. 2 | drawn by hand; counts from the manifest `downstream/npj/ref_standard.py` writes |
| Extended Data Fig. 3 | drawn by hand; counts from `registry/registry_estimate.py` |
| Extended Data Fig. 4 | `figures/ed2_cells_v16.py`, from `registry/registry_estimate.py` |
| Extended Data Fig. 5 | `figures/panel_prompt_sensitivity.py`, from `downstream/analyze_goal5_robustness.py` |
| Extended Data Fig. 6 | `figures/fig4_v15.py`, from `figures/evid_by_site.py` and `figures/panel_trigger_section.py` |
| Extended Data Fig. 7 | `figures/ed_registry.py`, from `registry/treatment_inputs.py` |
| Extended Data Table 1 | `downstream/build_table1.py` |
| Extended Data Table 2 | `registry/registry_estimate.py` (shares and gaps), `registry/treatment_inputs.py` (follow-up) |

### Supplementary Information

| item | script |
|---|---|
| Supplementary Note 1 | `downstream/npj/f01_restricted.py`, `registry/verification_independent.py` |
| Supplementary Note 2 | `registry/stratum_decomposition.py` |
| Supplementary Note 3 | `skill/aud-adjudication/SKILL.md` |
| Supplementary Fig. 1 | `figures/supp_era_v9.py` |
| Supplementary Fig. 2 | `figures/panel_registry_disparity.py`, from `registry/registry_estimate.py` |
| Supplementary Fig. 3 | `figures/panel_f_robustness.py`, from `downstream/npj/f04_model_disparity.py` and `f04_protocol_disparity.py` |
| Supplementary Fig. 4 | `figures/supp_medication_before_diagnosis.py`, counts from `registry/medication_before_t0.py` |
| Supplementary Table 1 | `downstream/npj/supp_confusion_six.py` |
| Supplementary Table 2 | `downstream/npj/f01_accuracy_against_reference.py` |
| Supplementary Table 3 | `downstream/npj/supp_table8.py` |
| Supplementary Table 4 | `registry/registry_estimate.py`, `registry/publish_tables.py` |
| Supplementary Table 5 | `registry/registry_estimate.py` |
| Supplementary Table 6 | `downstream/npj/f04_subgroup_disparity.py`, `f04_supp_tables.py` |
| Supplementary Table 7 | `downstream/site_figure_data.py`, `downstream/npj/f04_subgroup_rates.py` |
| Supplementary Table 8 | `downstream/npj/site_transfer.py` |
| Supplementary Table 9 | `downstream/npj/f06_figure6.py`, `f04_supp_tables.py` |
| Supplementary Table 10 | `registry/registry_estimate.py`, `downstream/npj/f04_supp_tables.py` |
| Supplementary Table 11 | `downstream/npj/f04_model_disparity.py`, `f04_protocol_disparity.py`, `f04_supp_tables.py` |
| Supplementary Tables 12, 13 | `downstream/npj/f04_figure4.py` |
| Supplementary Table 17 | `downstream/npj/f01_agreement_definitions.py` |
| Supplementary Tables 14, 19 | descriptive, written by hand |
| Supplementary Table 18 | `downstream/npj/build_supp_table6.py`, from `downstream/structured_baselines.py`, `note_ml_baseline.py` and `note_bert_baseline.py` |
| Supplementary Table 15 | the code sets in `downstream/registry_frame.py` |
| Supplementary Table 16 | `registry/sensitivity_rows.py` |

## 6. The statistics

```
downstream/stats_core.py    the estimators
downstream/data.yaml        the scale and how it collapses, the engines, the patient groups
```

One module owns the estimators: Cohen's and weighted kappa, Gwet's AC1, Wilson
intervals, percentile and paired bootstrap, McNemar's test, DeLong's test and
Benjamini–Hochberg. Its self-test runs on synthetic data and needs no study data:

```sh
cd downstream && python3 -c "import stats_core; stats_core._selftest()"
```

An indeterminate output counts against the classifier that gave it. Bootstrap intervals
use 2,000 resamples, except `f01_agreement_definitions.py` (4,000) and the cumulative
incidence draws and the band of Fig. 2d in `registry/` (500).

## 7. Not here

- **The data**: rating files, the registry tables, the note text.
- **The engine serving, the harness and the note extraction.** The skill, the text
  every engine was given, is in `skill/` and in Supplementary Note 3.
- **The chart-review application** the hepatologists used.
- **The page layout of the figures**, done by hand in a slide deck from the panels.
