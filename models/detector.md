# Egg Detector

## Role

Detect every egg in one controlled-lighting, top-down tray image, with support
for 0–5 eggs. The detector must provide stable bounding boxes for downstream
damage, cleanliness, and geometry measurements.

## Current state

- Default artifact: `egg_detection/finetune_egg_v1/weights/best.pt`
- Baseline artifact: `egg_detection/train1/weights/best.pt`
- Framework: YOLOv8 (`yolov8s.pt` fine-tuned)
- Active fine-tune dataset: one class (`egg`) under `data/detection_finetune/`
- Dataset split: Train 169 images/333 boxes, Validation 23/60, Test 19/57
- Audit: `reports/detection_finetune_dataset_report.json` is valid; one orphan
  validation label was quarantined, and exact/near-duplicate/oversized checks
  currently report zero.
- Locked test at confidence 0.75: precision 0.9584, recall 1.0000, mAP50
  0.9860, mAP50-95 0.9135, exact tray-count accuracy 0.8947, 0 missed eggs per
  tray, 0.1579 false positives per tray.
- 1-5 egg smoke subset: 17 images/41 eggs, exact count 0.9412, 0 missed eggs
  per tray, 0.0588 false positives per tray, mAP50 0.9881. This is a demo check,
  not the 0-5 beta gate.
- 0-egg trays are not present in the current test data.

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

- [ ] Create immutable train/validation/test manifests with source-level splits
  (image-level manifest exists; source metadata is insufficient).
- [ ] Audit labels for box tightness, duplicate boxes, missing eggs, and class
  consistency (visual check still pending).
- [x] Confirm the detector class represents egg presence; damage is classified
  by the damage model rather than conflated with detection.
- [ ] Retrain with the final static-tray dataset and fixed reproducibility
  settings.
- [ ] Tune augmentation only on training data.
- [x] Select and document the production confidence threshold (0.75) from
  validation precision/recall tradeoffs.
- [x] Add a test report with mAP, per-class metrics, missed eggs per tray,
  false positives per tray, and count accuracy.
- [ ] Add inference tests for 0–5 detections and invalid/empty images.
- [ ] Export and checksum the release checkpoint.

Visual verification of the one-class labels is still pending. The detector also
fails the release count/FP thresholds on the full test split, has no 0-egg tray
case, and its test split has only 19 images. Do not treat the current detector
metrics as release evidence.
