#!/usr/bin/env bash
set -euo pipefail

# Make all repository-relative paths work regardless of the caller's cwd.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MANIFEST_DIR="${MANIFEST_DIR:-data/damage/manifests}"
CHECKPOINT="${CHECKPOINT:-models/egg_grader.pth}"
METRICS="${METRICS:-models/damage_metrics.json}"
REPORT="${REPORT:-reports/damage_test_report.json}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"

python src/verify_damage_dataset.py \
  --manifest-dir "$MANIFEST_DIR" \
  --image-root "$ROOT_DIR"

PYTHONPATH=src python src/train_classifier.py \
  --labels "$MANIFEST_DIR/train.csv" \
  --val-labels "$MANIFEST_DIR/val.csv" \
  --image-root . \
  --output "$CHECKPOINT" \
  --metrics "$METRICS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed 42

mkdir -p "$(dirname "$REPORT")"
PYTHONPATH=src python src/evaluate_classifier.py \
  --labels "$MANIFEST_DIR/test.csv" \
  --image-root . \
  --checkpoint "$CHECKPOINT" \
  --batch-size "$BATCH_SIZE" \
  --output "$REPORT"

echo "Damage baseline complete."
