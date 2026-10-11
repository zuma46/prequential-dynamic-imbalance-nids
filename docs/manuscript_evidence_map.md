# Manuscript evidence map

Associated manuscript: **Prequential Evaluation of Dynamic Class Imbalance and Minority-Class Failure in Network Intrusion Detection**

This document maps the manuscript research questions to the corresponding
machine-readable evidence in the repository. It does not redefine the
scientific analysis.

## RQ1: dynamic class imbalance and attack-class lifecycle

**Question.** How is dynamic class imbalance manifested through temporal class support and attack-class lifecycle in chronological network-flow streams?

Primary repository evidence includes:

- `results/authoritative_final_analysis/02_tables/table_class_lifecycle.csv`
- `results/authoritative_final_analysis/02_tables/table_dynamic_imbalance_summary.csv`
- `results/authoritative_final_analysis/02_tables/table_lifecycle_events.csv`
- `results/authoritative_final_analysis/02_tables/table_support_threshold_sensitivity.csv`
- `results/authoritative_final_analysis/02_tables/table_window_sensitivity.csv`
- `results/manuscript_reference/table_class_lifecycle.csv`
- `results/manuscript_reference/table_lifecycle_events.csv`
- `results/manuscript_reference/Table_S_50k_Support_Adequacy.csv`

Interpretation boundary: disappearance, recurrence, minority switching, local imbalance, and adjacent-block distributional change describe observed temporal class dynamics. They are not by themselves a concept-drift oracle.

## RQ2: support/minority-state association, degradation, and complete failure

**Question.** How are within-class temporal support and dynamic-minority state associated with per-class recall, and where do support-qualified minority degradation and complete failure emerge?

Primary repository evidence includes:

- `analysis/relationship_v1_1/run_e1_rq1_formal_relationship_analysis.py`
- `analysis/relationship_v1_1/RQ1_FORMAL_RELATIONSHIP_PROTOCOL_LOCK_v1_0.json`
- `results/relationship_v1_1_reference/Table_SR1_Primary_Within_Class_Relationships.csv`
- `results/relationship_v1_1_reference/Table_SR2_Primary_Treatment_Coefficient_Contrasts.csv`
- `results/relationship_v1_1_reference/Table_SR3_Relationship_Sensitivity_Summary.csv`
- `results/manuscript_reference/Table_SR1_Primary_Within_Class_Relationships.csv`
- `results/manuscript_reference/Table_SR2_Primary_Treatment_Coefficient_Contrasts.csv`
- `results/manuscript_reference/Table_SR3_Relationship_Sensitivity_Summary.csv`
- `results/manuscript_reference/Table_S_Minority_Failure_Degradation.csv`
- `results/manuscript_reference/Table_S_DMAR_Threshold_BlockValues.csv`
- `results/manuscript_reference/Table_S_DMAR_Threshold_Sensitivity.csv`

The primary support floor is 50 observations, with 10 and 100 as sensitivity floors. Relationship coefficients are within-class temporal associations and are not causal effects.

## RQ3: controlled online imbalance-treatment comparison

**Question.** How do controlled online imbalance treatments differ in minority-sensitive performance and in the observed within-class support–performance relationships over time?

Primary repository evidence includes:

- `results/authoritative_final_analysis/02_tables/table_final_cumulative_model_summary.csv`
- `results/authoritative_final_analysis/02_tables/table_final_per_class_performance.csv`
- `results/authoritative_final_analysis/02_tables/table_primary_window_class_vulnerability.csv`
- `results/authoritative_final_analysis/02_tables/table_sampling_summary.csv`
- `results/authoritative_final_analysis/04_statistics/pairwise_moving_block_bootstrap.csv`
- `results/authoritative_final_analysis/04_statistics/temporal_model_ranks.csv`
- `results/manuscript_reference/Table_S_Complete_Per_Class_Performance.csv`
- `results/manuscript_reference/Table_S_Final_Cumulative_Per_Seed_Variability.csv`
- `results/manuscript_reference/Table_S_MOOB_Primary_Effects_With_WinFraction.csv`
- `results/manuscript_reference/Table_S_Pairwise_MBB_Full_Numeric.csv`
- `results/manuscript_reference/Table_S_Global72_Holm_Sensitivity.csv`
- `results/manuscript_reference/Table_S_Hyperparameter_Provenance.csv`

The controlled family supports mechanism attribution under a common Hoeffding-tree backbone. It does not support a universal state-of-the-art superiority claim.

## Input and preprocessing provenance

- `results/manuscript_reference/Table_S_49_Predictive_Feature_Contract.csv`
- `results/manuscript_reference/Table_S_Final_Input_Contract_By_Config.csv`
- `results/manuscript_reference/Table_S_Preprocessing_Source_Evidence.csv`
- `configs/`
- `chronology/PROVENANCE.md`
- `metadata/authoritative_hashes.json`
- `metadata/frozen_seeds.json`

## Reproduction and integrity

- `README.md`
- `docs/reproducibility.md`
- `docs/data_availability.md`
- `reproduce.sh`
- `PUBLIC_FILE_MANIFEST.csv`
- `SHA256SUMS.txt`
- `metadata/authoritative_hashes.json`
