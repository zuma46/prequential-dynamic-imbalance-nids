# E1 RQ1 Formal Relationship Analysis v1.0

- Authoritative input: `${E1_PROJECT_ROOT}/outputs/E1_FINAL_ANALYSIS_v1_0/01_master/nonoverlap_block_class_metrics_derived.parquet`
- SHA256: `f920643490e5d5441751c33536f262efd76655ed663b97255a3a6db485ccba67`
- Seed-averaged active attack class-block rows: 3,412
- Learner execution: NO
- Raw dataset read: NO
- Retuning: NO

## Primary relationship estimates (support floor 50, MBB L=20)

- NF-CSE-CIC-IDS2018-v3 | HT | support_slope: 0.2217 [0.1228, 0.3882], Holm=0.02220, status=OK
- NF-CSE-CIC-IDS2018-v3 | HT | minority_contrast: -0.2594 [-0.4730, -0.1123], Holm=0.02220, status=OK
- NF-CSE-CIC-IDS2018-v3 | OB | support_slope: 0.2556 [0.1622, 0.3917], Holm=0.00480, status=OK
- NF-CSE-CIC-IDS2018-v3 | OB | minority_contrast: -0.2528 [-0.4429, -0.1209], Holm=0.01680, status=OK
- NF-CSE-CIC-IDS2018-v3 | MOOB | support_slope: 0.2136 [0.1075, 0.3866], Holm=0.02220, status=OK
- NF-CSE-CIC-IDS2018-v3 | MOOB | minority_contrast: -0.2540 [-0.3911, -0.1532], Holm=0.01100, status=OK
- NF-CSE-CIC-IDS2018-v3 | MUOB | support_slope: 0.0000 [0.0000, 0.0000], Holm=NA, status=DEGENERATE_CONSTANT_RECALL
- NF-CSE-CIC-IDS2018-v3 | MUOB | minority_contrast: 0.0000 [0.0000, 0.0000], Holm=NA, status=DEGENERATE_CONSTANT_RECALL
- NF-ToN-IoT-v3 | HT | support_slope: 0.2989 [0.1668, 0.3543], Holm=0.00120, status=OK
- NF-ToN-IoT-v3 | HT | minority_contrast: -0.5526 [-0.7604, -0.2407], Holm=0.00160, status=OK
- NF-ToN-IoT-v3 | OB | support_slope: 0.2968 [0.1603, 0.3594], Holm=0.00120, status=OK
- NF-ToN-IoT-v3 | OB | minority_contrast: -0.5590 [-0.7682, -0.2425], Holm=0.00160, status=OK
- NF-ToN-IoT-v3 | MOOB | support_slope: 0.0928 [-0.0423, 0.1642], Holm=0.15317, status=OK
- NF-ToN-IoT-v3 | MOOB | minority_contrast: -0.1460 [-0.3567, 0.1453], Holm=0.27540, status=OK
- NF-ToN-IoT-v3 | MUOB | support_slope: 0.0000 [0.0000, 0.0000], Holm=NA, status=DEGENERATE_CONSTANT_RECALL
- NF-ToN-IoT-v3 | MUOB | minority_contrast: 0.0000 [0.0000, 0.0000], Holm=NA, status=DEGENERATE_CONSTANT_RECALL

## Secondary treatment-coefficient contrasts vs HT

- NF-CSE-CIC-IDS2018-v3 | OB-HT | support_slope: 0.0340 [-0.0501, 0.0934], Holm=1.00000
- NF-CSE-CIC-IDS2018-v3 | MOOB-HT | support_slope: -0.0081 [-0.1405, 0.1397], Holm=1.00000
- NF-CSE-CIC-IDS2018-v3 | OB-HT | minority_contrast: 0.0066 [-0.0931, 0.1155], Holm=1.00000
- NF-CSE-CIC-IDS2018-v3 | MOOB-HT | minority_contrast: 0.0054 [-0.1658, 0.2435], Holm=1.00000
- NF-ToN-IoT-v3 | OB-HT | support_slope: -0.0021 [-0.0243, 0.0199], Holm=1.00000
- NF-ToN-IoT-v3 | MOOB-HT | support_slope: -0.2060 [-0.2441, -0.1494], Holm=0.00400
- NF-ToN-IoT-v3 | OB-HT | minority_contrast: -0.0065 [-0.0437, 0.0334], Holm=1.00000
- NF-ToN-IoT-v3 | MOOB-HT | minority_contrast: 0.4065 [0.2736, 0.4986], Holm=0.00540

## Interpretation boundary

These are within-class temporal associations derived from frozen class-block evidence. They are not causal effects and do not establish concept drift.
