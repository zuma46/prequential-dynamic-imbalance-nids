# E1-TDIM-MSB Final Analysis Summary

Generated: 2026-08-05T15:47:55.798787+00:00

## Analysis integrity

- Final experimental runs consumed: **32/32**.
- NF-CSE evidence is the temporally corrected v1.1 full rerun; NF-ToN evidence is the validated v1.0 run reused by the v1.1 manifest.
- Non-overlapping analysis block: **50,000 flows**.
- Primary temporal visualization window: **500,000 flows**.
- Overlapping rolling windows are used for temporal visualization and vulnerability summaries, not as independent observations for significance tests.
- Primary pairwise inference uses paired moving-block bootstrap on non-overlapping 50k blocks, block length=20 blocks (1,000,000 flows), 5,000 replicates, followed by Holm adjustment.

## NF-CSE-CIC-IDS2018-v3

- Best final cumulative Macro-F1: **MOOB (0.757907)**.
- Best final cumulative Balanced Accuracy: **MOOB (0.754672)**.
- Best final cumulative Multiclass G-Mean: **MOOB (0.592051)**.
- Best final cumulative Dynamic Minority Attack Recall: **MOOB (0.737284)**.
- Best final cumulative Worst-Attack Recall: **MOOB (0.073924)**.
- Best final cumulative Benign FPR: **MUOB (0.000000)**.
- Attack-family lifecycle events: disappearance=15, recurrence=1, minority switching=14.
- Highest mean throughput: **MUOB (16941.6 instances/s)**.

## NF-ToN-IoT-v3

- Best final cumulative Macro-F1: **MOOB (0.801154)**.
- Best final cumulative Balanced Accuracy: **MOOB (0.780188)**.
- Best final cumulative Multiclass G-Mean: **MOOB (0.630827)**.
- Best final cumulative Dynamic Minority Attack Recall: **MOOB (0.703976)**.
- Best final cumulative Worst-Attack Recall: **MOOB (0.079245)**.
- Best final cumulative Benign FPR: **MUOB (0.000000)**.
- Attack-family lifecycle events: disappearance=16, recurrence=8, minority switching=10.
- Highest mean throughput: **MUOB (15706.7 instances/s)**.

## Statistical interpretation

- Holm-significant moving-block bootstrap comparisons: **54/72** pairwise tests.
- A positive `mean_advantage_a_over_b` means model A is better after respecting metric direction. For benign FPR, lower raw values are better.
- `supplementary_friedman.csv` is retained only as supplementary evidence because temporal autocorrelation is not modeled by the classical Friedman test.

## Generated outputs

- `02_tables/`: final model, per-class, lifecycle, window-sensitivity, and efficiency tables.
- `03_figures/`: prevalence, imbalance, prior-shift, rolling-performance, class-recall, and throughput figures.
- `04_statistics/`: moving-block bootstrap pairwise tests, temporal ranks, and supplementary Friedman results.
- `01_master/`: Parquet master files for reproducible secondary analysis.

This report summarizes generated evidence and statistical outputs.
Interpret the results together with the claim boundaries documented in
`docs/claim_boundaries.md`.