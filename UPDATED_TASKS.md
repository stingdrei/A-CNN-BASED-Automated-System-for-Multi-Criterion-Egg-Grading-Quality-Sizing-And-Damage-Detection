# UPDATED TASKS — Egg Grading System

Gap analysis against `egg-grading-system.skill` spec. Decisions applied.
Generated: 2026-09-22

---

## Decisions (Locked In)

| Decision | Choice | Source |
|----------|--------|--------|
| Backend framework | FastAPI (not Flask) | User override — project already uses FastAPI |
| Stained egg grading | Auto-Reject (Option A) | User choice — strictest rule |
| Size classification | PNS/BAFS 35:2005 mapped to mm | Philippine national standard |
| Dead code removal | Remove `live_view.py`, `roboflow_detect.py`, tracking config | User confirmed |
| Size categories | 3-class: Small / Medium / Large | Thesis scope simplification |
| Confidence threshold | Fix to 0.75 (single value, no range) | Spec requirement |

---

## Open Questions (Still Need Answers)

| # | Question | Status | Blocks |
|---|----------|--------|--------|
| Q1 | ~~Stained egg grading~~ | ✅ Resolved — Auto-Reject | — |
| Q2 | ~~Size bins~~ | ✅ Resolved — PNS/BAFS 35:2005 | — |
| Q3 | Dataset specs | Recommendation given, user must confirm final counts | P4.1 |
| Q4 | ~~Dead code~~ | ✅ Resolved — Remove | — |
| Q5 | Weight formula `k` constant | Needs empirical fitting from calibration set | P2.2 |
| Q6 | "Grade" vs "Class" terminology | Pick one, use everywhere | P5.2 |

---

## P1 — Critical Blockers

Nothing works without these. Fix first.

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| P1.1 | Create `backend/app/models.py` — all backend imports from it but file doesn't exist. Define User, Prediction, DetectionBox SQLAlchemy models | New: `backend/app/models.py` | ✅ Done |
| P1.2 | Fix MODEL_PATH in config — currently hardcoded to wrong directory `/Users/wii/Projects/python/egg-cv/...` | `backend/app/config.py` | ✅ Done |
| P1.3 | Remove tracking code — `src/live_view.py` has ObjectTracker class (centroid tracking for video). Spec says static tray only | Delete: `src/live_view.py` | ✅ Done |
| P1.4 | Remove roboflow_detect.py — dead code, not part of pipeline | Delete: `src/roboflow_detect.py` | ✅ Done |
| P1.5 | Remove tracking section from config.yaml — `tracking.max_distance` and `max_disappeared` params | `config/config.yaml` | ✅ Done |

---

## P2 — Grading Logic (Spec Compliance)

Grading rules are wrong or incomplete. Must fix before anything downstream works.

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| P2.1 | Fix size bins — current `<50/<60/else` doesn't match PNS/BAFS 35:2005. Use: Small <53mm, Medium 53-60mm, Large >60mm | `backend/app/ml/yolo_inference.py` → `calculate_size_category()` | ⬜ Todo |
| P2.2 | Fix weight formula — current `0.0005 * d³` is wrong. Use `W = k × L × B²` where k is empirically fit. Add placeholder k=0.52 with TODO to calibrate | `backend/app/ml/yolo_inference.py` → `estimate_weight()` | ⬜ Todo |
| P2.3 | Add L×B measurement — weight formula needs major axis (L) and minor axis (B), not just diameter. Update detection post-processing to extract both | `backend/app/ml/yolo_inference.py` → `_parse_results()` | ⬜ Todo |
| P2.4 | Implement grading rules — current `map_to_grade` only uses damage + size. Add cleanliness branch: Stained → auto-Reject | `backend/app/ml/yolo_inference.py` → `map_to_grade()` | ⬜ Todo |
| P2.5 | Fix per-egg JSON shape — current response missing `egg_id`, `cleanliness_status`, confidence sub-fields. Spec shape: `{egg_id, grade, size, weight_g, damage_status, cleanliness_status, confidence: {detection, damage, cleanliness}}` | `yolo_inference.py`, `schemas/__init__.py`, API response | ⬜ Todo |
| P2.6 | Update grading-rules.md — resolve the open question: stained eggs auto-Reject alongside damaged | `references/grading-rules.md` | ⬜ Todo |

---

## P3 — Cleanliness Classifier (Missing Feature)

Cleanliness is named in the thesis title but never implemented. Third classifier head required.

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| P3.1 | Create cleanliness classifier model — binary CNN: Clean vs Stained. Can reuse `EggGradingCNN` architecture with num_classes=2, or add as second head | New: `src/model_cleanliness.py` or extend `src/model.py` | ⬜ Todo |
| P3.2 | Define cleanliness dataset requirements — images labeled Clean vs Stained (dirt, fecal matter, blood spots). Min 150 per class | Document in `references/` | ⬜ Todo |
| P3.3 | Add cleanliness training pipeline — separate training script or extend `src/train.py` to handle cleanliness labels | `src/train.py` or new `src/train_cleanliness.py` | ⬜ Todo |
| P3.4 | Integrate cleanliness into inference — load cleanliness model, run on cropped egg, return status in response | `backend/app/ml/yolo_inference.py`, `backend/app/services/prediction_service.py` | ⬜ Todo |
| P3.5 | Add cleanliness confidence to response JSON — `confidence.cleanliness` field | `yolo_inference.py`, schemas | ⬜ Todo |

---

## P4 — Data Pipeline (CSV/DB Drift)

CSV and DB fields don't match spec. Pipeline incomplete.

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| P4.1 | Align CSV fields with spec — rename `class` → `damage_status`, add `cleanliness_status`, `tray_id`, `captured_at`. Update CSV header | `egg_statistics.csv` format + writer code | ⬜ Todo |
| P4.2 | Complete DB schema — add `trays` table (id, captured_at, image_ref, operator_id), ensure `eggs` table has all per-egg fields | `backend/app/models.py` (from P1.1), alembic migration | ⬜ Todo |
| P4.3 | Add CSV logger to inference pipeline — CSV exists but nothing writes to it. Create service that appends one row per detected egg | New service or update `backend/app/services/prediction_service.py` | ⬜ Todo |
| P4.4 | Match API endpoints to spec — `POST /api/grade`, `GET /api/records`, `GET /api/stats` | `backend/app/routers/predictions.py`, `backend/app/routers/dashboard.py` | ⬜ Todo |
| P4.5 | Create tray-level response — spec says return one JSON per detected egg in a tray, with tray metadata | `backend/app/routers/predictions.py` | ⬜ Todo |

---

## P5 — Training Pipeline (Model Quality)

Models need work to meet spec requirements.

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| P5.1 | Define final dataset specs — confirm image count, source, labeling protocol for damage + cleanliness + size. Document in Methods chapter | User decision → document in `references/` | ⬜ Todo |
| P5.2 | Create weight calibration script — fit `k` constant against eggs with known actual weights. Report fitted k + R² | New: `src/calibrate_weight.py` | ⬜ Todo |
| P5.3 | Update training config — ensure `config.yaml` reflects actual dataset paths, batch sizes, epochs for both damage and cleanliness models | `config/config.yaml` | ⬜ Todo |
| P5.4 | Add validation metrics — current training loop only prints loss. Add accuracy, precision, recall, F1 per epoch | `src/train.py` | ⬜ Todo |
| P5.5 | Fix egg_statistics.csv sample data — current file has 586 rows of bad data (all Large/80g/Reject). Clean or replace with real inference output | `egg_statistics.csv` | ⬜ Todo |

---

## P6 — Cleanup & Consistency

Polish and align everything.

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| P6.1 | Fix confidence threshold — pick 0.75, remove range comments, use consistently everywhere | `config/config.yaml`, `yolo_inference.py`, frontend | ⬜ Todo |
| P6.2 | Pick terminology — "Grade A/B/C" or "Class A/B/C"? Use one term throughout code + thesis | All code + docs | ⬜ Todo |
| P6.3 | Update egg-grading-system.skill — change stack from Flask to FastAPI, update size bins to PNS/BAFS, add decision for stained eggs | `egg-grading-system.skill` | ⬜ Todo |
| P6.4 | Remove config.yaml comment about tracking being "for continuous video" — system is static tray only | `config/config.yaml` | ⬜ Todo |
| P6.5 | Verify frontend matches API response shape — ensure React components handle new JSON fields (cleanliness_status, confidence sub-objects) | `frontend/src/pages/Result.tsx`, `frontend/src/types/` | ⬜ Todo |

---

## Execution Order

```
Phase 1 (Critical):  P1.1 → P1.2 → P1.3 → P1.4 → P1.5
Phase 2 (Grading):   P2.1 → P2.2 → P2.3 → P2.4 → P2.5 → P2.6
Phase 3 (Cleanliness): P3.1 → P3.2 → P3.3 → P3.4 → P3.5
Phase 4 (Data):      P4.1 → P4.2 → P4.3 → P4.4 → P4.5
Phase 5 (Training):  P5.1 → P5.2 → P5.3 → P5.4 → P5.5
Phase 6 (Cleanup):   P6.1 → P6.2 → P6.3 → P6.4 → P6.5
```

**Dependencies:**
- P2.4 (grading rules) needs P3.4 (cleanliness integrated) before it works end-to-end
- P4.2 (DB schema) needs P1.1 (models.py) first
- P5.1 (dataset specs) should be resolved before P3.2 (cleanliness dataset)
- P6.5 (frontend) should come after P2.5 (JSON shape) is finalized

---

## Reference: PNS/BAFS 35:2005 Size Classes

For code reference when implementing P2.1:

| Size Class | Weight Range (g) | Approx Length (mm) |
|------------|------------------|-------------------|
| Jumbo | ≥ 70 | > 65 |
| Extra-Large | 65–70 | 60–65 |
| Large | 60–65 | 57–62 |
| Medium | 55–60 | 53–58 |
| Small | 50–55 | 50–55 |
| Pullets | 45–50 | 47–52 |
| Peewee | 40–45 | 43–48 |

**Your 3-class mapping:**
- Small: < 53mm (covers Peewee + Pullets + Small)
- Medium: 53–60mm (covers Medium + Large)
- Large: > 60mm (covers Extra-Large + Jumbo)

---

## Reference: Current Grading Rule (To Be Updated)

```
if damage_status == "Damaged":
    grade = "Reject"
elif cleanliness_status == "Stained":    # <-- NEW, not yet implemented
    grade = "Reject"                      # <-- Decision: Option A
elif size == "Large":
    grade = "A"
elif size == "Medium":
    grade = "B"
elif size == "Small":
    grade = "C"
```
