# Size Estimator

## Role

Estimate egg size from calibrated geometry and map it to the project’s
three-class PNS/BAFS 35:2005 simplification:

- Small: `< 53 mm`
- Medium: `53–60 mm`
- Large: `> 60 mm`

These boundaries must be verified against the cited standard and the project’s
chosen measurement convention before release.

## Current state

The backend uses the correct `<53 / 53–60 / >60 mm` class boundaries but still
derives axes from the detection bounding box, marked `bbox_fallback`. This is
not a sufficiently precise measurement method for the final model. A physical
reference object and camera calibration are still required.

## Requirements

- Use a calibrated pixel-to-millimeter relationship for the actual camera,
  height, lens, and imaging plane.
- Extract major axis `L` and minor axis `B` from a segmentation mask, contour,
  or a validated ellipse fit; do not use only the maximum box side.
- Define behavior for partial, overlapping, blurred, or non-ellipse-shaped
  detections.
- Validate axis measurements against manually measured reference dimensions.
- Report millimeter error and per-class confusion on a frozen test set.
- Version calibration parameters with the camera/setup and inference result.
- Do not silently classify an out-of-range or low-quality measurement.

## TODO

- [ ] Confirm the exact PNS/BAFS measurement and size-class interpretation.
- [ ] Select a contour, segmentation, or ellipse-fitting method.
- [ ] Build a calibration dataset with known reference dimensions.
- [ ] Measure pixel-to-mm error across the tray field, not only at the center.
- [ ] Implement major/minor axis extraction with quality checks.
- [x] Replace the old diameter thresholds in backend inference.
- [ ] Evaluate size MAE, per-class recall, macro F1, and edge cases near
  53 mm and 60 mm.
- [ ] Store `length_mm`, `breadth_mm`, `size`, calibration version, and quality
  flags in the per-egg result.
