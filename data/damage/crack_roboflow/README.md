# Crack-only damage dataset

This task predicts shell crack presence only:

- `Damaged` (`0`): Roboflow row has `crack=1`.
- `Not Damaged` (`1`): row has no crack and is labeled `clean`, `dirty`, or both.

Cleanliness is not part of this classifier. In the inspected export, all 70
`clean+dirty` rows have `crack=0`, so they map to `Not Damaged` under this
definition.

## Data provenance and splits

The source is Roboflow project `eggs-msws9/training-qy3a3`, export dated
2026-11-24, licensed CC BY 4.0. Original 3,481 images remain in the external
source export; this repository contains only manifests and audit, not those
images. The source export had no validation/test folders.

`manifests/{train,validation,test}.csv` are deterministic, grouped splits
(seed `20261124`). The grouping proxy removes Roboflow's `.rf.<hash>` suffix so
variants of same base filename cannot cross splits. Audit lists exact image and
group counts, class balance, and label CSV hash. Test split is held out from
model fitting, but remains from same source project—not independent camera
data.

Rebuild manifests from the original export:

```bash
python3 scripts/prepare_crack_roboflow.py \
  --source /path/to/training \
  --output data/damage/crack_roboflow/manifests
```

## Train

Run from repository root. `--image-root` must point to directory containing the
export's `train/` folder; manifests use paths like `train/<image>.jpg`.

```bash
PYTHONPATH=src python3 src/train_classifier.py \
  --labels data/damage/crack_roboflow/manifests/train.csv \
  --val-labels data/damage/crack_roboflow/manifests/validation.csv \
  --image-root /path/to/training \
  --output models/crack_roboflow_resnet18.pth \
  --metrics reports/crack_roboflow_metrics.json \
  --class-names 'Damaged,Not Damaged' --arch resnet18 --pretrained \
  --stage1-epochs 5 --stage2-epochs 15 \
  --stage1-lr 1e-3 --stage2-lr 1e-4 --seed 20261124
```

Evaluate the test split only after selecting model/settings using validation:

```bash
PYTHONPATH=src python3 src/evaluate_classifier.py \
  --labels data/damage/crack_roboflow/manifests/test.csv \
  --image-root /path/to/training \
  --checkpoint models/crack_roboflow_resnet18.pth \
  --output reports/crack_roboflow_test.json
```

If transferring data to another PC, transfer the complete `training/` folder,
including `train/_classes.csv`, and use that folder for `--image-root`. Keep
this dataset's crack-only labels separate from cleanliness data and any future
real-camera test set.
