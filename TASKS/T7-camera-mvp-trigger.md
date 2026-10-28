# T7 — Camera MVP Implementation Trigger

## Purpose

Gate for implementing the **browser-camera one-shot capture loop**:

```
press Capture -> grab 1 frame -> detect -> grade all eggs -> display -> next tray
```

Do **not** start implementation until every trigger condition below is true.
This keeps UI work from racing ahead of an unproven model.

> **Beta override:** the browser camera UI was implemented on explicit user
> request before these gates passed. It is not release approval; the trigger
> remains closed and physical camera/model acceptance is still required.

## Trigger Conditions (all required)

### 1. Multi-criterion grading is implemented
- [x] Damage classification wired into inference.
- [ ] Cleanliness classification wired into inference (`Clean` / `Stained`).
- [ ] Size class computed from calibrated major axis (`<53 / 53–60 / >60` mm).
- [ ] Weight computed as `W = k × L × B²` with fitted `k` (not `null`).
- [x] Deterministic grading rules applied:
  `Damaged -> Reject`, `Stained -> Reject`, `Clean + Large/Medium/Small -> A/B/C`,
  unknown/low-confidence -> `N/A` (cleanliness unavailable -> final `N/A`).

### 2. Per-egg output contract is complete
- [x] Every detection returns:
  `egg_id`, `grade`, `size`, `weight_g`, `damage_status`,
  `cleanliness_status`, `confidence.{detection,damage,cleanliness}`
  (`weight_g` may be null until calibrated; `cleanliness_status=Unavailable`).
- [x] API, CSV, database, and frontend use the **same field names**.
- [x] No damaged or stained egg can receive `A`, `B`, or `C`.

### 3. "True results" are proven on a locked test set
Reports committed under `reports/`, evaluated once on a frozen egg-level set:

- [x] Detector: mAP50 >= 0.98, recall >= 0.99 (full 19-image/57-egg test).
- [ ] Detector: exact tray-count accuracy >= 0.95 on **0–5 egg** trays.
- [ ] Detector: <= 0.10 false positives per tray; <= 0.02 missed eggs per tray.
- [ ] Damage: balanced accuracy >= 0.93; damaged recall >= 0.98.
      Numeric result at validation-selected 0.18: balanced 0.9792, damaged
      recall 1.0000. Keep unchecked until source leakage is resolved and the
      test is rerun under a single locked protocol.
- [ ] Cleanliness: balanced accuracy >= 0.95; stained recall >= 0.98.
- [ ] Size: macro F1 >= 0.90; MAE <= 1.5 mm.
- [ ] Weight: MAE <= 3 g; R² >= 0.85; >= 90% within ±3 g.
- [ ] End-to-end: exact grade accuracy >= 0.95; reject recall >= 0.98;
      false-accept rate <= 1%.
- [ ] 0-egg and 5-egg trays explicitly tested.

### Physical tray evidence command

```bash
python src/validate_physical_trays.py \
  --images data/physical_trays \
  --expectations data/physical_tray_expectations.csv \
  --output reports/physical_tray_validation.json \
  --csv reports/physical_tray_validation.csv \
  --fail-on-gate
```

Expectations CSV: `filename,expected_count,notes`. The tool requires at least
one 0-egg and one 5-egg tray, exact count accuracy `>=0.95`, false positives
`<=0.10/tray`, and missed eggs `<=0.02/tray`.

### 4. "90%+ done" is defined by the gates, not a vibe
Treat "model is 90%+ done" as: **sections 1–3 all checked**, plus:
- [ ] Every model has a frozen test report and reproducible checkpoint.
- [ ] No train/validation/test leakage.
- [ ] Thresholds selected on validation only.
- [ ] Model versions, thresholds, calibration version stored per inference batch.

If any item above is unchecked, the trigger is **closed**.

## Implementation Status (Beta Override)

- [x] `CameraCapture` starts the rear camera, previews it, captures one JPEG,
  and stops tracks on unmount.
- [x] Permission/API errors produce readable messages and a retry action.
- [x] Upload page supports `/upload?mode=camera` and file upload tabs.
- [x] Result page has **Next Tray** returning to camera mode.
- [x] Backend has a 0-detection completion test.
- [ ] Real browser/camera test on HTTPS.
- [ ] Physical 5-egg tray returns exactly 5 rows.
- [ ] Manual double-submit and camera-permission-denied test.

## Trigger Action

When all conditions pass, implement the browser-camera MVP:

### Flow
```
/upload?mode=camera
  -> camera preview (getUserMedia, facingMode=environment)
  -> CAPTURE button
  -> canvas.toBlob('image/jpeg', 0.92)
  -> POST /api/v1/predictions/upload
  -> /result/:id
  -> "Next Tray" -> back to camera
```

### Scope
| File | Change |
|---|---|
| `frontend/src/components/upload/CameraCapture.tsx` | new: preview, capture, retake, stop tracks on unmount |
| `frontend/src/pages/Upload.tsx` | Capture / Upload tabs |
| `frontend/src/pages/Result.tsx` | "Next Tray" button; per-egg damage/cleanliness columns |
| `frontend/src/types/index.ts` | add damage/cleanliness/confidence fields |
| `frontend/src/api/client.ts` | send `confidence_threshold`/`save_annotated` as query params |

### Acceptance
- [ ] Capture one frame -> one prediction -> N per-egg results.
- [ ] 5-egg tray -> exactly 5 rows.
- [ ] 0-egg tray -> 0 rows, no error.
- [ ] "Next Tray" reopens camera; no double-submit.
- [ ] Camera permission denied -> readable error.
- [ ] Existing file-upload flow unchanged.
- [ ] Requires HTTPS or `localhost`.

## Non-goals (until MVP passes)

- Live auto-capture / conveyor mode.
- Frame tracking or video IDs.
- Multi-tray queueing.

Live auto-trigger comes **after** the MVP, using the same capture + grade
functions with a tray-present/stability/cooldown state machine.

## Readiness Check

```bash
# Model/test evidence
ls reports/damage_test_report.json reports/detector_finetune_test_report.json
ls reports/cleanliness_test_report.json reports/size_test_report.json \
   reports/weight_calibration_report.json reports/end_to_end_test_report.json

# Contract + service tests
./.venv/bin/python -m unittest discover -s tests -q

# Backend boots and models load
./.venv/bin/uvicorn app.main:app --app-dir backend --reload
```

If any report or test is missing, keep this trigger closed.
