# Damage Classifier Error Audit — Baseline

Source: `reports/damage_error_audit.csv` (120 test samples)
Checkpoint: `models/egg_grader.pth`
Label convention: `0 = Damaged`, `1 = Not Damaged`
Confidence = softmax probability of the predicted class.

## Confusion counts

| error_type | count |
|---|---|
| TP (Damaged → Damaged) | 87 |
| TN (Not Damaged → Not Damaged) | 17 |
| FN (Damaged → Not Damaged) | 8 |
| FP (Not Damaged → Damaged) | 8 |

Matches baseline report `reports/damage_baseline_report.json`
(accuracy 0.8667, damaged recall 0.9158, confusion `[[87, 8], [8, 17]]`).

## False negatives — damaged eggs called "Not Damaged" (highest priority)

| filename | confidence (misclassified as Not Damaged) |
|---|---|
| data/damage/Damaged/damaged_13.jpg | 0.9666 |
| data/damage/Damaged/damaged_146.jpg | 0.8704 |
| data/damage/Damaged/damaged_465.jpg | 0.9040 |
| data/damage/Damaged/damaged_457.jpg | 0.8505 |
| data/damage/Damaged/damaged_123.jpg | 0.7964 |
| data/damage/Damaged/damaged_185.jpg | 0.7581 |
| data/damage/Damaged/damaged_249.jpg | 0.5876 |
| data/damage/Damaged/damaged_517.jpg | 0.5350 |

## False positives — clean eggs called "Damaged"

| filename | confidence (misclassified as Damaged) |
|---|---|
| data/damage/Not Damaged/not_damaged_112.jpg | 0.9998 |
| data/damage/Not Damaged/not_damaged_62.jpg | 0.9838 |
| data/damage/Not Damaged/not_damaged_142.jpg | 0.8104 |
| data/damage/Not Damaged/not_damaged_110.jpg | 0.7973 |
| data/damage/Not Damaged/not_damaged_116.jpg | 0.7160 |
| data/damage/Not Damaged/not_damaged_80.jpg | 0.6396 |
| data/damage/Not Damaged/not_damaged_42.jpg | 0.5994 |
| data/damage/Not Damaged/not_damaged_87.jpg | 0.5066 |

## Decision required

Label corrections are irreversible. Inspect the filenames above and decide
whether any labels (or prompt quality) should be fixed before Phase 4 training.
No manifest has been modified by this audit.
