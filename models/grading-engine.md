# Grading Engine and Pipeline Integration

## Role

Combine detector, damage, cleanliness, and size outputs into one deterministic
per-egg grade. The engine is not a learned model, but it is part of the final
model system and must be tested as strictly as the learned components.

## Locked grading policy

```text
if damage_status == "Damaged":
    grade = "Reject"
elif cleanliness_status == "Stained":
    grade = "Reject"
elif size == "Large":
    grade = "A"
elif size == "Medium":
    grade = "B"
elif size == "Small":
    grade = "C"
else:
    grade = "N/A"
```

Unknown, invalid, or below-threshold inputs must not receive A, B, or C.

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

- [ ] Implement the locked cleanliness branch in the backend.
- [ ] Finalize the per-egg JSON schema and use it everywhere.
- [ ] Add exhaustive grading-rule unit tests.
- [ ] Add end-to-end tests for damaged, stained, clean-small,
  clean-medium, and clean-large eggs.
- [ ] Add low-confidence and invalid-input behavior.
- [ ] Remove stale `AA` and incompatible grade terminology from frontend types.
- [ ] Evaluate complete-tray grading accuracy on the frozen test set.
- [ ] Add model and calibration metadata to persisted predictions.
