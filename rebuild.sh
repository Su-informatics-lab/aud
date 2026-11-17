#!/bin/bash
# Rebuild the results of the paper, from the hepatologists' classifications and the per-patient
# model classifications onward. Run from the analysis root on the cluster, where the inputs
# listed in provenance.md live; nothing here runs without them.
set -o pipefail
export PYTHONPATH=figures:downstream/npj:downstream
run() { echo "== $*"; "$@" || echo "FAILED: $*"; }

# ---- 1. Ground truth, agreement, accuracy, patient groups, comparators ----
run python3 downstream/npj/registry_frame_ids.py
run python3 downstream/npj/ref_standard.py
# The registry table comes first: downstream/aud_dx.py reads the AUD diagnosis (registry flag) from it.
run python3 registry/source_table.py
run python3 downstream/build_rater_table.py
run python3 downstream/analyze_goal5_robustness.py
run python3 downstream/build_npj_materials.py
run python3 downstream/npj/f00_cohort_flow_and_table1.py
run python3 downstream/npj/f01_accuracy_against_reference.py
run python3 downstream/npj/panels_de.py
run python3 downstream/npj/f01_agreement_definitions.py
run python3 downstream/npj/f01_kappa_inventory.py
run python3 downstream/npj/f01_restricted.py
run python3 downstream/npj/f01_primary_migration.py
run python3 downstream/npj/f01_supp_temporal.py
run python3 downstream/npj/f03_evidence_categories.py
run python3 downstream/npj/f04_figure4.py
run python3 downstream/npj/f04_model_disparity.py
run python3 downstream/npj/f04_protocol_disparity.py
run python3 downstream/npj/f04_subgroup_rates.py
run python3 downstream/npj/f04_subgroup_disparity.py
run python3 downstream/npj/f06_subgroup_frame.py
run python3 downstream/npj/f04_subgroup_feasibility.py
run python3 downstream/cci_fairness.py
run python3 downstream/npj/f05_figure5.py
run python3 downstream/run_parity.py
run python3 downstream/npj/f04_supp_documentation.py
run python3 downstream/npj/f06_figure6.py
run python3 downstream/structured_baselines.py
run python3 downstream/npj/site_xgb.py
run python3 downstream/npj/site_transfer.py
run python3 downstream/npj/build_supp_table6.py
run python3 downstream/npj/supp_table8.py
run python3 downstream/site_figure_data.py
run python3 downstream/npj/f02_supp_era.py
run python3 downstream/build_table1.py
run python3 downstream/npj/supp_confusion_six.py
# Run separately. The quotation audit reads the note text; the note comparators need a batch
# job (note_ml_baseline.py exceeds the login-node time limit, note_bert_baseline.py needs a GPU).
#   python3 downstream/run_npj_provenance.py && python3 downstream/grounding_finish.py
#   python3 downstream/note_ml_baseline.py
#   python3 downstream/note_bert_baseline.py

# ---- 2. Registry estimate, treatment, sensitivity analyses ----
run python3 registry/registry_estimate.py 2000
run python3 registry/diagnosis_accuracy_and_curves.py
run python3 registry/treatment_inputs.py
run python3 registry/meddx_estimate_and_band.py
run python3 registry/drug_extraction_check.py
run python3 registry/verification_independent.py
run python3 registry/stratum_decomposition.py
run python3 registry/sensitivity_rows.py
run python3 registry/predictive_values.py
run python3 registry/t0_summary.py
run python3 registry/medication_before_t0.py
run python3 registry/publish_tables.py
run python3 downstream/npj/f04_supp_tables.py

# ---- 3. Figure panels ----
O=figures/out; F=output/registry
mkdir -p $O/fig1 $O/fig2 $O/fig3 $O/fig4 $O/ed $O/sf $O/new $O/stability
run python3 figures/panel_model_leaderboard.py --out $O/fig4
run python3 figures/panel_audit_v8.py --out $O/fig4
run python3 figures/panel_trigger_section.py --out $O/fig4
run python3 figures/stability.py
run python3 figures/evid_by_site.py
run python3 figures/acc_readers.py
for p in panel_b panel_d panel_e panel_f panel_g_v15; do run python3 figures/$p.py --out $O/fig1; done
run python3 figures/fig2nh_ab.py --out $O/fig2
run python3 figures/fig2nh_de_v16.py --json $F/master_numbers_2026-10-26.json \
    --aj $F/aj_curves_by_stratum_2026-10-26.csv --band $F/share_band_2026-10-26.csv --out $O/fig2
for p in panel_subgroup_disparity panel_roc_space panel_site panel_transfer; do run python3 figures/$p.py --out $O/fig3; done
run python3 figures/panel_fig3_d.py
run python3 figures/fig4_v15.py
run python3 figures/ed2_cells_v16.py --json $F/master_numbers_2026-10-26.json --out $O/ed
run python3 figures/ed_registry.py --cif $F/cif_crude_2026-10-26.csv \
    --prior $F/prior_users_by_stratum_2026-10-26.csv --counts $F/aj_by_label_summary_2026-10-26.csv --out $O/ed
run python3 figures/panel_prompt_sensitivity.py --out $O/ed
run python3 figures/supp_era_v9.py --out $O/sf
run python3 figures/panel_registry_disparity.py --out $O/sf
run python3 figures/panel_f_robustness.py
run python3 figures/supp_medication_before_diagnosis.py
