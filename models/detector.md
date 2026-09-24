# Egg Detector

## Role

Detect every egg in one controlled-lighting, top-down tray image, with support
for 0–5 eggs. The detector must provide stable bounding boxes for downstream
damage, cleanliness, and geometry measurements.

## Current state

- Artifact: `models/egg_detection_finetuned/weights/best.pt`
- Framework: YOLO
- Dataset class: `egg`; damage is handled by the downstream classifier
- The checked-in checkpoint predates this correction and must be retrained
  before deployment.
- Training log: 50 epochs in `models/egg_detection_finetuned/results.csv`
- Best logged mAP@50: approximately 0.835 at epoch 46
- Final logged epoch: precision approximately 0.740, recall approximately 0.750,
  mAP@50 approximately 0.815, mAP@50–95 approximately 0.616

These are baseline training-run metrics, not a final independent test result.

## Requirements

- Use the static-tray task only; do not use tracking.
- Include empty trays and trays with 1–5 eggs in validation and test data.
- Ensure every egg is annotated consistently, including touching or partially
  occluded eggs.
- Evaluate missed eggs per tray, false positives per tray, and exact count
  accuracy in addition to mAP.
- Select the production confidence threshold on validation data; keep it fixed
  during testing and deployment.
- Preserve sufficient box quality for contour/ellipse measurement, not merely
  object presence.
- Test different shell colors, orientations, tray positions, lighting levels,
  and camera distances.

## TODO

- [ ] Create immutable train/validation/test manifests with source-level splits.
- [ ] Audit labels for box tightness, duplicate boxes, missing eggs, and class
  consistency.
- [x] Confirm the detector class represents egg presence; damage is classified
  by the damage model rather than conflated with detection.
- [ ] Retrain with the final static-tray dataset and fixed reproducibility
  settings.
- [ ] Tune augmentation only on training data.
- [ ] Select and document the production confidence threshold from validation
  precision/recall tradeoffs.
- [ ] Add a test report with mAP, per-class metrics, missed eggs per tray,
  false positives per tray, and count accuracy.
- [ ] Add inference tests for 0–5 detections and invalid/empty images.
- [ ] Export and checksum the release checkpoint.

The dataset audit currently reports 298 oversized boxes. These annotations
must be visually reviewed and corrected before the detector metrics can be
treated as release evidence.
