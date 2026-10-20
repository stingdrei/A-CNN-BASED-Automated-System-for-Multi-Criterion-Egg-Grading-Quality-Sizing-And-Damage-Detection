# Grading Rules — Locked Specification

Authoritative rules for the static, top-down tray grading pipeline. Any change
here must be mirrored in `src/grading.py` and the backend inference service.

## Inputs

For a single tray image (0–5 eggs), each detected egg yields:

- damage status + confidence
- cleanliness status + confidence
- calibrated major axis `L` (mm) and minor axis `B` (mm)
- detection confidence

## Size classification (PNS/BAFS)

Based on the calibrated major axis `L`:

| Size | Rule |
|---|---|
| Small | `L < 53 mm` |
| Medium | `53 mm ≤ L ≤ 60 mm` |
| Large | `L > 60 mm` |

Invalid / non-positive `L` → `Unknown`.

## Weight estimation

```
W = k × L × B²
```

- `k` is fitted from known-weight calibration data before production use.
- Invalid inputs (`L ≤ 0`, `B ≤ 0`, `k ≤ 0`) → no estimate.
- Report fitted `k` with R², MAE, RMSE, signed error, and % within ±3 g.

## Grading rules

| Condition | Grade |
|---|---|
| Damaged | Reject |
| Stained | Reject |
| Clean + Large | A |
| Clean + Medium | B |
| Clean + Small | C |
| Unknown / invalid / low-confidence | N/A |

No damaged or stained egg may receive A, B, or C. Unknown or low-confidence
results are surfaced as `N/A` (or `Needs Review` where the backend requires an
operator decision) — never silently converted into a passing grade.

## Per-egg output contract

```json
{
  "egg_id": 1,
  "grade": "A",
  "size": "Large",
  "weight_g": 68.5,
  "damage_status": "Not Damaged",
  "cleanliness_status": "Clean",
  "confidence": {
    "detection": 0.82,
    "damage": 0.91,
    "cleanliness": 0.88
  }
}
```

The API, CSV (`egg_statistics.csv`), database, and frontend must use these
identical field names.

## Label / status conventions

- Damage classifier labels: `0 = Damaged`, `1 = Not Damaged`.
- Class order in checkpoints: `("Damaged", "Not Damaged")`.
- Detector has a single class: `egg` (damage classified downstream).
- Terminology: use `Grade A/B/C` everywhere; legacy `AA` is not supported.

## Confidence thresholds

- Detector: `0.75` (locked target).
- Damage review: `0.75` → below threshold routes to `Needs Review`.
