# Final Model Plan — Egg Grading System

## Objective

Build a reproducible, high-accuracy pipeline that grades a **single static,
top-down tray image containing 0–5 eggs** under controlled lighting. The system
must report one result per detected egg:

- damage status and confidence
- cleanliness status and confidence
- calibrated size class
- calibrated weight estimate
- final grade
- detection confidence

The system must not claim final accuracy until the complete pipeline has been
evaluated on a locked, egg-level test set that was not used for training,
augmentation, threshold selection, or calibration fitting.

## Current model inventory

| Component | Current artifact | Current role | Readiness |
|---|---|---|---|
| Egg detector | `egg_detection/finetune_egg_v1/weights/best.pt` | Locate eggs in a tray image | Baseline fine-tune; needs controlled retraining and locked test evaluation |
| Damage classifier | `models/egg_grader_crop_resnet18_v1.pth` | Damaged / Not Damaged on per-egg crops | Trained and integrated in the backend; metrics in `reports/damage_crop_resnet18_v1_test_report.json` |
| Cleanliness classifier | `models/egg_grader_cleanliness_v1.pth` | Clean / Stained | Beta web/stock candidate; not release-approved (`CLEANLINESS_RELEASE_APPROVED=false`) |
| Size estimator | Geometry in `src/grading.py` + `backend/app/ml/yolo_inference.py` | Small / Medium / Large from calibrated pixels | Implemented; gated off until `CALIBRATION_VERSION` is set |
| Weight estimator | `estimate_weight` in `src/grading.py` | Reported grams from calibrated geometry | Implemented; returns nothing until `WEIGHT_K` is fitted |
| Grading engine | `backend/app/services/grading_service.py` | Combine all criteria into grade | Implemented; size-derived letters suppressed while uncalibrated |

## Target architecture

```text
tray image
  -> YOLO egg detection
  -> per-egg crop with padding and bounds checks
  -> damage classifier
  -> cleanliness classifier
  -> ellipse/contour geometry + camera calibration
  -> size class and weight estimate
  -> deterministic grading rules
  -> API / database / CSV / dashboard
```

The pipeline is image-based and static. Do not add frame tracking, FPS targets,
or video-only IDs to the final implementation.

## System-wide requirements

### Dataset and split discipline

- Maintain independent train, validation, and test sets at the **egg/source
  level**, not only at the image level. Images or frames from the same egg,
  capture session, or video must not cross splits.
- Record source, capture conditions, egg identifier, labeler, label date, and
  disagreement status for every sample.
- Freeze the test set before threshold tuning.
- Use class-balanced sampling or explicit class weights where imbalance exists.
- Keep all preprocessing and augmentation deterministic for evaluation.
- Use realistic holdout conditions: lighting variation, camera distance,
  orientation, shell color, occlusion, and tray position.

### Metrics required before release

Report metrics overall and by relevant subgroup:

- detection: precision, recall, mAP@50, mAP@50–95, false detections per tray,
  missed eggs per tray, and exact tray-count accuracy
- damage and cleanliness: accuracy, balanced accuracy, precision, recall, F1,
  specificity, confusion matrix, and calibration/reliability
- size: macro F1, per-class recall, confusion matrix, and mean absolute error
  in millimeters
- weight: MAE, RMSE, mean signed error, percentage within ±3 g, and fitted
  calibration constant with R²
- end-to-end: per-egg exact grading accuracy, reject recall, false-accept rate,
  and complete-tray success rate

Do not replace per-class and end-to-end metrics with a single accuracy number.

### Release gates

The final model is not release-ready until:

1. Every model has a frozen test report and reproducible checkpoint.
2. No train/validation/test leakage is found.
3. Confidence thresholds are selected on validation data and evaluated once on
   the test set.
4. Inference handles 0–5 eggs, malformed images, empty crops, and partial
   detections explicitly.
5. The API, CSV, database, and frontend use the same per-egg field names.
6. Unknown/low-confidence results are surfaced rather than silently converted
   into a passing grade.
7. Model version, calibration version, threshold configuration, and timestamp
   are stored with every inference batch.

## Inference contract

Each detected egg should expose this logical shape:

```json
{
  "egg_id": 1,
  "grade": "A",
  "size": "Large",
  "weight_g": 68.5,
  "damage_status": "Not Damaged",
  "cleanliness_status": "Clean",
  "confidence": {
    "detection": 0.82,
    "damage": 0.91,
    "cleanliness": 0.88
  }
}
```

The system must also preserve the raw geometry and model versions internally
for auditability.

## Required implementation order

1. Establish dataset manifests and leakage checks.
2. Retrain and evaluate the detector on the static-tray task.
3. Evaluate and, if necessary, retrain the damage classifier.
4. Collect, label, train, and evaluate the cleanliness classifier.
5. Implement calibrated major/minor axis measurement and size classification.
6. Fit and validate the weight constant using known-weight eggs.
7. Finalize grading rules and confidence/abstention behavior.
8. Integrate the complete pipeline and run end-to-end test evaluation.
9. Version the release artifacts and publish the test report.

See the individual plans in `MODELS/`.

## Immediate damage-model workflow

The current source images are in `data/damage/`. The old
`data/processed/` generated manifests were removed because they referenced
missing augmented files. Create new manifests, then train from the repository
root:

```bash
python3 src/create_damage_split.py
PYTHONPATH=src python3 src/train_classifier.py \
  --labels data/damage/manifests/train.csv \
  --val-labels data/damage/manifests/val.csv \
  --image-root . \
  --output models/egg_grader.pth \
  --metrics models/damage_metrics.json
```

Yes, train a model after creating the split. The first run is a damage-model
baseline, not the final high-accuracy release: evaluate it on the untouched
test manifest, inspect false negatives, and only then tune architecture,
augmentation, and thresholds. Do not use the test set to make training
decisions.
