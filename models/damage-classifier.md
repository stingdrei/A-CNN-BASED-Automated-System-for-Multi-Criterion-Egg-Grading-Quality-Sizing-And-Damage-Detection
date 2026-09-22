# Damage Classifier

## Role

Classify each detector crop as `Damaged` or `Not Damaged`. Damage includes the
damage categories defined by the labeling protocol; the protocol must state
whether cracks, holes, shell breakage, and subtle hairline defects are included.

## Current state

- Architecture: `src/model.py`, `EggGradingCNN`
- Input: 224×224 RGB crop
- Classes: 2
- Checkpoint: `models/egg_grader.pth`
- Dataset folders currently contain 632 damaged and 162 not-damaged images,
  which is materially imbalanced.
- The training script reports only the last batch loss and does not save
  validation metrics.

The checkpoint is not sufficient evidence of high accuracy until a reproducible
test report is generated.

The supported source dataset is `data/damage/`. Generate manifests
from those class folders; do not train from the removed `data/processed`
augmented manifests.

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
- [ ] Build source-level train/validation/test manifests.
- [ ] Add class-balanced sampling or weighted loss.
- [ ] Upgrade the training loop to track loss, accuracy, balanced accuracy,
  precision, recall, F1, and AUROC per epoch.
- [ ] Add early stopping, best-checkpoint selection, and reproducible seeds.
- [ ] Evaluate the current checkpoint on a frozen test set.
- [ ] Tune the damage threshold for reject recall on validation data.
- [ ] Compare the current CNN with a transfer-learning model.
- [ ] Integrate the selected classifier into FastAPI inference.
- [ ] Store damage confidence and model version with each egg result.
