# Cleanliness Classifier

## Role

Classify each egg crop as `Clean` or `Stained`. Stains include the agreed
categories of dirt, fecal matter, and blood spots. The labeling protocol must
define borderline cases and whether small cosmetic marks count as stained.

## Current state

No cleanliness checkpoint, dataset, training script, or inference integration
exists. This is a required model, not an optional enhancement.

## Requirements

- Collect representative clean and stained eggs under the same imaging setup as
  deployment.
- Use at least 150 labeled samples per class as a minimum starting point; use a
  larger balanced dataset for the final model where possible.
- Capture stain size, location, shell color, glare, shadows, and difficult
  borderline examples.
- Use at least two independent labelers for a subset and record agreement.
- Split by physical egg/source, not by near-duplicate image.
- Report balanced accuracy, precision, recall, specificity, F1, confusion
  matrix, and calibration.
- Treat a stained prediction as `Reject` under the locked project decision.
  Optimize to avoid false-clean predictions.
- Add an abstention threshold for ambiguous glare, blur, and occlusion.
- Store the cleanliness class order and preprocessing with the checkpoint.

## TODO

- [ ] Finalize the cleanliness labeling guide and examples.
- [ ] Collect and label clean/stained crops with source metadata.
- [ ] Measure inter-labeler agreement and adjudicate disagreements.
- [ ] Create leakage-safe train/validation/test manifests.
- [ ] Implement a binary training pipeline with class balancing and
  reproducibility controls.
- [ ] Compare a transfer-learning baseline with the shared CNN architecture.
- [ ] Tune the threshold to protect stained-egg recall.
- [ ] Freeze and checksum the selected checkpoint.
- [ ] Integrate inference and return `cleanliness_status` plus confidence.
- [ ] Add end-to-end tests proving stained eggs cannot receive A/B/C.
