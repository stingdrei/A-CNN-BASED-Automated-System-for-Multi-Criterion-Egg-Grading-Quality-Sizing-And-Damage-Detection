# Reviewed cleanliness dataset

Frozen, manually reviewed crops from `good-dirty-egg-crops`. Use only `train.csv`
and `valid.csv` for the first training experiment. Class IDs are `Clean=0` and
`Stained=1`.

## Contents and provenance

- `source/accepted_manifest.csv`: accepted crop decisions, original source labels,
  quality flags, source groups, split assignments, SHA-256, and image checks.
- `source/review.csv.snapshot`: frozen source review table (line endings normalized
  for portability). Does not alter original labels or live review CSV; original
  source-file SHA-256 is recorded in the audit.
- `source/audit.md` and `source/audit.json`: counts and integrity report.
- `source/real_camera_test_collection_template.csv`: empty collection scaffold,
  **not** test data.
- `manifests/{train,valid}.csv`: ready for `src/train_classifier.py`.

Crops with `small_for_model=true` or `edge_touch=true` remain in the source
manifest as `quality_holdout`; they are excluded from train/validation until
separately adjudicated. Reviewed labels remain unchanged. Source groups do not
cross train/validation.

No independently labeled real-camera test set is included. Collect and label
100 distinct clean and 100 distinct stained physical eggs separately; keep all
views of an egg/session together and do not use this set for model selection.

## Put crop files in place on training machine

Copy the `unverified/` directory from the data-preparation machine into
`data/reviewed_cleanliness/crops/unverified/` in this repository. Do not copy
shortlist images as the real-camera test set. Example from a machine that can
read the source data:

```bash
mkdir -p data/reviewed_cleanliness/crops/unverified
rsync -a /path/to/good-dirty-egg-crops/unverified/ \
  data/reviewed_cleanliness/crops/unverified/
```

Verify exported manifests from the committed source manifest with:

```bash
python3 scripts/export_reviewed_cleanliness.py \
  --source data/reviewed_cleanliness/source/accepted_manifest.csv \
  --output-dir data/reviewed_cleanliness/manifests
```

## First training experiment (not run here)

```bash
PYTHONPATH=src python3 src/train_classifier.py \
  --labels data/reviewed_cleanliness/manifests/train.csv \
  --val-labels data/reviewed_cleanliness/manifests/valid.csv \
  --image-root data/reviewed_cleanliness \
  --output models/cleanliness_reviewed_resnet18.pth \
  --metrics reports/cleanliness_reviewed_metrics.json \
  --class-names Clean,Stained --arch resnet18 --pretrained \
  --stage1-epochs 5 --stage2-epochs 15 \
  --stage1-lr 1e-3 --stage2-lr 1e-4 --seed 20261124
```

This evaluates only on the grouped validation split. Do not use the independent
real-camera set until model selection is complete.
