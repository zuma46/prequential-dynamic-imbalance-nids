# Formal Within-Class Relationship Analysis

This directory contains the implementation used to quantify temporal
within-class associations between attack-class support, dynamic-minority
state, and per-class recall.

## Scientific scope

The analysis is read-only and operates on the derived exact-block
class-performance table. It does not train or update HT, OB, MOOB, or MUOB,
does not modify model states, and does not retune experimental parameters.

## Primary analyses

1. Within-class support-recall association using a class-fixed-effect
   coefficient on log10(class support).
2. Within-class dynamic-minority-state recall contrast.
3. OB-minus-HT and MOOB-minus-HT contrasts of the relationship coefficients.

Primary support floor: 50 observations.

Support-floor sensitivity:
- 10
- 50
- 100

Primary moving-block-bootstrap span:
- 20 exact 50,000-flow blocks

Sensitivity spans:
- 10
- 20
- 40 blocks

Bootstrap replicates:
- 5,000

Multiplicity adjustment:
- Holm adjustment within dataset.

MUOB is retained descriptively when attack-class recall is degenerate.

## Reproduction

Use the repository entrypoint:

```bash
bash reproduce.sh relationship \
  --class-block /path/to/nonoverlap_block_class_metrics_derived.parquet \
  --out ./relationship_results