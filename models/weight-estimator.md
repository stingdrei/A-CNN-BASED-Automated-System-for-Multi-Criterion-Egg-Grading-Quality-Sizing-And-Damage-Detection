# Weight Estimator

## Role

Estimate and report weight in grams from calibrated egg geometry. Weight is a
reported field and is not currently a grading criterion.

## Formula

```text
W = k × L × B²
```

- `L`: major axis in millimeters
- `B`: minor axis/breadth in millimeters
- `k`: fitted constant from a calibration set of eggs with known actual weights

Do not hardcode the historical `0.05 × D³` or `0.0005 × D³` formulas as the
final method. A placeholder `k` may be used only during development and must be
clearly marked as uncalibrated.

Fit `k` with `src/calibrate_weight.py` after collecting a CSV with
`length_mm`, `breadth_mm`, and `actual_weight_g`:

```bash
python src/calibrate_weight.py     --data data/weight_calibration.csv     --output reports/weight_calibration_report.json
```

The backend uses `WEIGHT_K=0.0` until the fit is accepted. Set `WEIGHT_K` and
`CALIBRATION_VERSION` in the environment to activate calibrated estimates.

## Requirements

- Weigh calibration eggs with a validated scale and record actual grams.
- Cover the expected range of egg sizes, shell colors, and shapes.
- Keep calibration eggs separate from the final weight test set.
- Fit `k` using the same geometry and camera setup used in deployment.
- Report fitted `k`, R², MAE, RMSE, signed error, and percentage within ±3 g.
- Validate calibration at multiple tray positions and camera sessions.
- Surface estimates outside the validated range as low confidence rather than
  presenting false precision.

## TODO

- [ ] Collect paired `(L, B, actual_weight_g)` calibration records.
- [x] Implement a fitting script at `src/calibrate_weight.py`.
- [ ] Compare a single global `k` with a validated regression alternative.
- [x] Report fitted `k`, confidence interval, R², MAE, RMSE, and ±3 g rate.
- [ ] Add calibration version and sample count to model metadata.
- [x] Replace the current diameter-cubed implementation with `k × L × B²`.
- [x] Add unit tests for the fit, zero axes, invalid rows, and weight alias.
- [ ] Validate weight estimates on a locked test set before deployment.
