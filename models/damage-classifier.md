# Damage Classifier

## Role

Classify each detector crop as `Damaged` or `Not Damaged`. Damage includes the
damage categories defined by the labeling protocol; the protocol must state
whether cracks, holes, shell breakage, and subtle hairline defects are included.

## Current state

- Architecture: `custom` (`EggGradingCNN`) or `resnet18`, resolved through
  `src/model_factory.py`; backend reads architecture metadata from the checkpoint.
- Input: 224×224 RGB crop
- Classes: 2 (`Damaged`, `Not Damaged`)
- Default backend checkpoint: `models/egg_grader_crop_resnet18_v1.pth`
  (ResNet18, 43 MB). It stores `arch`, `class_names`, `img_size`, and
  `damaged_threshold=0.18`.
- Legacy checkpoint: `models/egg_grader.pth` (custom full-image classifier).
- v2 manifests: 1163 images found, 4 exact duplicate files (2 conflicting label
  groups) removed, 1159 used; split 811/174/174.
- Source groups: 632 legacy damaged, 162 legacy not-damaged, 369 broken-egg
  images from one capture session. Strict source/session split is unverified.
- Detector-crop train/val/test: 571/128/132 usable crops (357/78/84 damaged).
- Crop coverage: train 726/811 detected, 571 usable, 155 multi-detection
  skipped, 85 missed; val 162/174 detected, 128 usable, 34 skipped, 12 missed;
  test 160/174 detected, 132 usable, 28 skipped, 14 missed.

| Evaluation | Accuracy | Balanced | Damaged recall |
| --- | ---: | ---: | ---: |
| Legacy full-image, v2 test (174) | 0.7816 | 0.6583 | 0.9833 |
| Legacy full-image, crop test (132) | 0.3939 | 0.3720 | 0.4524 |
| Custom crop fine-tune, crop test (132) | 0.8788 | 0.8869 | 0.8571 |
| ResNet18 crop, crop test (132) | 0.9848 | 0.9792 | 1.0000 |

ResNet18 crop validation (128): accuracy 0.9922, balanced 0.9900, damaged
recall 1.0000, confusion `[[78,0],[1,49]]`. Test at the validation-selected
threshold 0.18: confusion `[[84,0],[2,46]]`.

Threshold selection uses `src/select_damage_threshold.py` on validation only,
with constraints damaged recall >= 0.98, balanced accuracy >= 0.93, and
not-damaged recall >= 0.98. It selected 0.18, the lowest passing threshold.
The test set was inspected during earlier threshold exploration, so this is
provisional evidence, not a locked release proof. Reports:
`reports/damage_crop_resnet18_threshold_selection.json`,
`reports/damage_crop_val_resnet18_v1_report.json`,
`reports/damage_crop_classifier_resnet18_v1_test_report.json`,
`reports/damage_crop_resnet18_train_metrics.json`.

The supported source dataset is `data/damage/`. Use `data/damage/manifests_v2/`;
do not train from the removed `data/processed` augmented manifests.

## Requirements

- Use detector-derived crops for evaluation, while retaining a separately
  labeled crop test set for diagnosing crop errors.
- Balance classes through sampling, class-weighted loss, or an explicitly
  justified alternative.
- Keep eggs from the same source or video in one split.
- Report balanced accuracy, precision, recall, specificity, F1, confusion
  matrix, and probability calibration.
- Prioritize high recall for damage: a damaged egg must not silently become a
  passing grade.
- Define an abstention/low-confidence path instead of forcing an uncertain
  crop into `Not Damaged`.
- Compare the current CNN against a stronger transfer-learning baseline before
  release; retain the simpler model only if it meets the test gates.
- Save the class order and preprocessing normalization with the checkpoint.

## TODO

- [ ] Audit the existing labels and document the damage taxonomy.
- [ ] Build source-level train/validation/test manifests (blocked by missing
  source/session metadata; v2 is best-effort image-level with exact duplicates removed).
- [x] Add class-balanced sampling or weighted loss.
- [x] Track loss, accuracy, balanced accuracy, macro F1, and both per-class
  recalls per epoch. AUROC remains TODO.
- [x] Select best checkpoints by balanced accuracy by default; `--select-metric`
  can choose macro F1, recall, or F1. The current ResNet18 checkpoint predates
  this fix and was selected by class-1 F1.
- [x] Evaluate the current checkpoint on a frozen test set.
- [x] Tune the damage threshold for reject recall on validation data.
- [x] Compare the current CNN with a transfer-learning model.
- [x] Integrate the selected classifier into FastAPI inference.
- [x] Store damage confidence and model version with each egg result.
