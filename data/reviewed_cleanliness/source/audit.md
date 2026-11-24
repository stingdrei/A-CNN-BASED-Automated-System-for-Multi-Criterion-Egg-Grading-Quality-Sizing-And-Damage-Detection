# Reviewed crop dataset audit

Snapshot: `review.csv.snapshot`
Review CSV SHA-256: `b2441f44b95651f0165da7440edd576cb261e7eac6662056bdd9d0d11f1d1ffe`

Accepted crops: **3061**
Training-eligible crops (unflagged): **2605**
Quality holdout (small or edge-touch): **456**
Source groups: **37**
Split seed: `20261124`; validation fraction target: 20%

## Accepted labels

| Label | Count |
|---|---:|
| Clean | 1522 |
| Stained | 1539 |

## Split

| Split | Clean | Stained | Source groups |
|---|---:|---:|---:|
| train | 877 | 1194 | 28 |
| validation | 268 | 266 | 9 |
| quality_holdout | 377 | 79 | 17 |
| unassigned | 0 | 0 | 0 |

## Integrity checks

- Missing crops: 0
- Unreadable crops: 0
- Crop dimension mismatches: 0
- Exact duplicate image groups: 0
- Groups crossing train/validation: 0
- Small-for-model flags: 310
- Edge-touch flags: 165
- Any quality flag is kept in `quality_holdout`, excluded from train/validation without changing its reviewed label.

`real_camera_test_collection_template.csv` is an empty 100-per-class collection scaffold, not test data.
`accepted_manifest.csv` preserves source labels and review decisions, adds split and integrity metadata.
No crops are moved, relabeled, or deleted. The real-camera independent test set is not present and must be collected/labeled separately; never use it for model selection.
