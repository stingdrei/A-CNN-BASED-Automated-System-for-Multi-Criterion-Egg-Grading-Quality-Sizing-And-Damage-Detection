# Grading Engine and Pipeline Integration

## Role

Combine detector, damage, cleanliness, and size outputs into one deterministic
per-egg grade. The engine is not a learned model, but it is part of the final
model system and must be tested as strictly as the learned components.

## Locked grading policy

```text
if damage_status == "Damaged" or cleanliness_status == "Stained":
    grade = "Reject"
elif damage_status == "Not Damaged" and cleanliness_status == "Clean":
    grade = {"Large": "A", "Medium": "B", "Small": "C"}.get(size, "N/A")
else:
    grade = "N/A"
```

Unknown, invalid, or below-threshold inputs must not receive A, B, or C.
Cleanliness is currently `Unavailable`, so final grades are `N/A` and the API
exposes `provisional_grade` from damage + size only.

Damage status is decided from `P(Damaged)` using a validation-selected
threshold stored in the checkpoint (`0.18` for the current ResNet18 crop
model). A damaged prediction is never downgraded to `Needs Review`: it rejects
even when its confidence is below the review threshold. Review applies only
when a not-damaged prediction is below the confidence threshold.

## Requirements

- Use one terminology choice consistently: `Grade A/B/C` plus `Reject`.
- Preserve damage, cleanliness, size, weight, and confidence as separate fields.
- Reject damaged or stained eggs regardless of size.
- Define an explicit low-confidence/needs-review state if any required input
  fails its release threshold.
- Keep the rule engine pure and unit-testable.
- Test all combinations, including unknown values and boundary sizes.
- Keep API, CSV, PostgreSQL, and frontend field names aligned.
- Record model versions, thresholds, calibration version, and timestamp.

## TODO

- [x] Implement the locked cleanliness branch in the backend (unavailable -> N/A).
- [x] Finalize the per-egg JSON schema and use it in API, DB, CSV, frontend.
- [x] Add exhaustive grading-rule unit tests.
- [ ] Add end-to-end tests for damaged, stained, clean-small,
  clean-medium, and clean-large eggs (stained/clean blocked by missing model).
- [x] Add low-confidence and invalid-input behavior.
- [x] Remove stale `AA` and incompatible grade terminology from frontend types.
- [ ] Evaluate complete-tray grading accuracy on the frozen test set.
- [x] Add model and calibration metadata to persisted predictions.
