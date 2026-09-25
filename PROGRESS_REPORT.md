# Progress Report

Date: 2026-09-25

## Detector Demonstration Findings

The current YOLO detector was trained using a dataset composed primarily of
single-egg images. This dataset does not represent the intended deployment
scenario: a tray image containing up to five eggs under varied lighting and
capture conditions.

During live-view testing, the detector showed the following behavior:

- At a confidence threshold of `0.75`, it usually detected only one or two
  eggs in a five-egg tray.
- At a confidence threshold of `0.50`, it sometimes detected three eggs when
  the eggs were well separated and the lighting and camera distance were
  favorable.
- In favorable conditions it occasionally detected all five eggs, but the
  detections were not consistently precise.
- The detector can therefore demonstrate partial multi-egg detection, but it
  cannot currently be presented as a reliable five-egg tray detector.

The problem is not only the confidence threshold. The current annotations are
heuristic single-image boxes rather than verified tray-level bounding boxes.
The audit identified 298 oversized boxes covering at least 75% of the image
area. On held-out single-egg images, the detector often returns broad
image-spanning boxes and may produce duplicate boxes. This indicates that the
model has not learned precise egg boundaries.

## Current Assessment

The trained model is a prototype and is not ready for final deployment or
accuracy claims. Its live-view behavior is useful for demonstrating progress,
but the results are limited by dataset mismatch, annotation quality, and
unrepresented tray conditions.

## Corrective Data-Collection Plan

The next work package is to collect a new detector dataset that matches the
intended task:

1. Capture images containing five eggs arranged on a tray, plus valid examples
   containing fewer eggs and an empty tray where appropriate.
2. Vary lighting, shadows, camera distance, camera angle, background, tray
   position, egg orientation, shell color, and partial overlap.
3. Record the damage and cleanliness classification for each egg so the
   detector and downstream classifiers can be aligned with the same samples.
4. Draw one tight, manually verified `egg` bounding box around every visible
   egg. Do not use damage status as detector class IDs.
5. Split images by tray, capture session, and source egg before training so
   related images cannot leak across train, validation, and test sets.
6. Freeze a representative tray-level test set and select the confidence
   threshold using validation data only.

After collection and annotation, the detector will be retrained with the
single `egg` class and evaluated using per-egg recall, precision, mAP,
missed-eggs-per-tray, false-positives-per-tray, and exact tray-count accuracy.
The current model will remain the baseline until the new model passes that
comparison.

## Current Status

### Completed milestones

- T1: Critical blockers addressed and validated.
  - Backend models were created for `User`, `Prediction`, and `DetectionBox`.
  - The model path was fixed to resolve from the repository root.
  - Video-tracking code was removed to align with the static-tray architecture.
  - Legacy detection scripts not used by the supported pipeline were removed.
  - Project configuration was simplified to remove outdated tracking parameters.
- Damage-model baseline run completed.
  - Train/validation/test manifests were generated with no filename overlap.
  - `models/egg_grader.pth` was trained with the reproducible shell workflow.
  - The untouched test set contains 120 samples and was evaluated independently.
  - Baseline results: 86.67% accuracy, 79.79% balanced accuracy, 79.79% macro-F1,
    and 91.58% damaged recall.
- Detector dataset preparation was validated.
  - After removing one source-conflicting duplicate from train and test, the
    detector dataset contains 554 train, 119 validation, and 119 test images.
  - Every image has a matching label file and all labels have valid YOLO coordinates.
  - `data/detection/data.yaml` now resolves correctly from repository-relative paths.
  - `src/train_yolo.py` now works independently of the caller's current directory
    and fails early when required image splits are missing.
  - `src/verify_detection_dataset.py` now writes a frozen-style CSV manifest and
    JSON audit report, including image hashes, class counts, and annotation source.
  - The validator now generates sampled box-overlay previews and reports
    oversized boxes for visual review; the current report contains 298 boxes
    covering at least 75% of their image area.
  - The audit found and removed one byte-identical cross-split duplicate with
    conflicting source labels: `train/damaged_155.jpg` and
    `test/not_damaged_150.jpg`.

### In progress / not yet complete

- T2: Grading logic and API contract
  - Size thresholds, calibration formula, cleanliness handling, and grading rules still need to be fully implemented and validated.
- T3: Cleanliness classifier
  - Dataset creation, training pipeline, inference integration, and test validation remain pending.
- T4: Data pipeline and persistence
  - CSV/database/API consistency, schema migrations, and tray/result integration still need work.
- T5: Training, calibration, and evaluation
  - The damage classifier has a reproducible baseline and independent test report.
  - Detector training and evaluation are still outstanding.
  - Final model artifacts, calibration analysis, and end-to-end evaluation are still outstanding.
- T6: Cleanup, consistency, and release
  - Release validation, documentation alignment, and final release preparation have not yet been completed.

## Overall Assessment

The project has reached a solid baseline state for the static-tray backend, damage classification, and detector data preparation. It is not yet a production-ready grading pipeline: grading rules, cleanliness classification, verified detector ground truth, calibration, persistence, and end-to-end release validation remain open.

## Key Risks and Dependencies

- Data pipeline and field-name consistency across CSV, database, API, and frontend outputs.
- Cleanliness classification must be trained and integrated before stained eggs can be reliably rejected.
- The imported `DIrty Egg` JSONL files contain classification/image transcripts rather than bounding-box annotations.
  The current detector labels therefore require manual verification before detector metrics can be trusted.
- The detector dataset is currently based on heuristic contour-generated boxes, not a verified annotation set.
- Detector manifest freezing is no longer blocked by exact or perceptual duplicate
  leakage; representative boxes still require visual verification.
- Weight calibration and size logic must be validated against project specifications before production use.
- Final release readiness depends on full end-to-end evaluation, documentation alignment, and consistency checks.

## Immediate Next Steps

1. Complete the grading rule and API contract updates in T2.
2. Resolve the duplicate-label leakage and verify detector bounding boxes, then
   manually train and evaluate the detector.
3. Prepare and train the cleanliness model defined in T3.
4. Align CSV/database/frontend persistence and tray-level outputs in T4.
5. Execute calibration, threshold selection, and end-to-end evaluation in T5.
6. Finalize cleanup and release validation in T6.

## Conclusion

The architecture is stabilized and the damage baseline is now reproducible, but the model inventory is incomplete. The remaining work is concentrated in verified detector annotations, cleanliness classification, grading/calibration integration, persistence consistency, and final release validation.
