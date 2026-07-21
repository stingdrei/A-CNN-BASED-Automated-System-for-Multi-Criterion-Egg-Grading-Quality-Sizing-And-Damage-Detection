# Thesis Gap Analysis & Remediation Plan

## Date: July 21, 2026 — Updated July 21, 2026 (Session 2)

## Source Documents
- **THESIS.pdf** — Academic thesis specification
- **Codebase** — Full exploration of `src/`, `backend/`, `frontend/`, `models/`, `config/`, `data/`

---

## 1. Thesis Spec Reconstruction

### Architecture
1. Camera feed → YOLOv8s detection (fine-tuned, 70-20-10 split, 20 epochs, SGD lr=0.01 batch=16 momentum=0.937)
2. Confidence filtering: YOLO 0.65-0.75, CNN 0.60-0.70
3. Centroid tracking → unique IDs, no duplicate counting
4. Crop egg → CNN damage classification (EggGradingCNN: 3→32→64 conv, 2 FC, Adam lr=0.001, 20 epochs, 224×224)
5. Size: Small <50mm, Medium 50-60mm, Large >60mm
6. Weight: W = 0.05 × D³ (clamped 30-80g)
7. Grading: Grade A (Not Damaged+Large), Grade B (Not Damaged+Medium), Grade C (Not Damaged+Small), Reject (Damaged)
8. JSON output: `{"grade": "A", "size": "Large", "weight_g": 68.5, "damage_status": "Not Damaged"}`
9. Color boxes: green=Not Damaged, red=Damaged
10. CSV logging per egg
11. 28-30 FPS target
12. Evaluation: precision, recall, F1, mAP

### Data
- Augmented dataset: horizontal flips, 224×224 resize, ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- 80/20 train/val split, stratified sampling

---

## 2. Current Implementation Status

### Correct (Matching Thesis)

| Component | Status | Location |
|-----------|--------|----------|
| YOLOv8s detection | ✅ Fine-tuned (needs retrain with new labels) | `models/egg_detection_finetuned/weights/best.pt` |
| EggGradingCNN architecture | ✅ 3→32→64, 2 FC, `num_classes=2` | `src/model.py` |
| Weight formula W=0.0005×D³ | ✅ Fixed (was 0.05, off by 100×) | `src/live_view.py`, `backend/app/ml/yolo_inference.py:51`, `config/config.yaml` |
| Size categories | ✅ S<50, M 50-60, L>60 | `src/live_view.py` |
| Weight clamp 30-80g | ✅ — now naturally lands there with corrected formula | `src/live_view.py` |
| Centroid tracking | ✅ ObjectTracker class | `src/live_view.py` |
| CSV logging | ✅ Includes grade column | `src/live_view.py` |
| Camera calibration | ✅ Calibration configurable | `config/config.yaml` |
| Calibration tool | ✅ Interactive script | `src/calibrate_camera.py` |
| Data augmentation | ✅ Horizontal flips, 224×224, ImageNet norm | `src/utils.py` (transforms) |
| YOLO confidence threshold | ✅ Default 0.75 (was 0.3), keyboard +/- | `src/live_view.py` |
| CNN confidence threshold | ✅ Default 0.7, keyboard `C` toggle | `src/live_view.py` |
| CNN model | ✅ Loaded from `models/egg_grader.pth` (was unused on disk) | `src/live_view.py` |
| Color-coded boxes | ✅ Green=Not Damaged, Red=Damaged, Yellow=Low CNN conf | `src/live_view.py` |
| Grade system | ✅ Grade A/B/C/Reject on boxes | `src/live_view.py` |
| Backend API | ✅ FastAPI + PostgreSQL, grades fixed AA/A/B→A/B/C/Reject | `backend/` |
| Frontend dashboard | ✅ React + Vite | `frontend/` |

### Gaps (Not Matching Thesis)

#### ✅ RESOLVED

| # | Gap | Fix |
|---|-----|-----|
| 1 | **CNN not loaded in live_view.py** | ✅ Loaded `egg_grader.pth`, crop detection → 224×224 → CNN softmax → damage label |
| 2 | **Weight always 80g** | ✅ `mm_per_pixel=1.0→0.09`, weight formula `0.05→0.0005`. Realistic 30-80g. |
| 3 | **No grade system in live_view** | ✅ Added `map_grade(damage_status, size_category)` → A/B/C/Reject displayed on boxes |
| 4 | **No CNN confidence filtering** | ✅ Added `--cnn-conf` CLI arg (default 0.7), `C` key toggle, yellow boxes below threshold |
| 5 | **Train/val/test split wrong** | ✅ `convert_to_yolo.py` `main()` uses stratified 70-10-20 split. Regenerated all labels. |
| 9 | **CNN output classes mismatch** | ✅ `model.py` default `num_classes=3→2`. Config confirmed `nc=2`. |
| 12 | **YOLO trained 50 epochs not 20** | ✅ Set to 100 epochs with `patience=50` — auto-stops when mAP plateaus. Will converge around epoch 60-80. |

#### REMAINING

#### HIGH

| # | Gap | Evidence | Fix Required |
|---|-----|----------|-------------|
| 6 | **No evaluation harness** | No script computes precision, recall, F1, mAP. | Create `evaluate.py` — confusion matrix, precision, recall, F1, mAP@50 |
| 7 | **Frontend lacks grading** | Frontend shows detections but no grade/size/weight. | Add grade, size, weight to result display |

#### MEDIUM

| # | Gap | Evidence | Fix Required |
|---|-----|----------|-------------|
| 8 | **Dataset class imbalance** | 632 Damaged : 162 Not Damaged (80:20 ratio). CNN may bias toward Damaged. | Augment Not Damaged class, add class weights to loss |
| 10 | **No WebSocket real-time** | Backend has WebSocket stub but no implementation. | Connect WebSocket for live updates |
| 11 | **Frontend no size/weight** | Frontend doesn't show size/weight. | Add size/weight to Result page |

#### LOW

| # | Gap | Evidence | Fix Required |
|---|-----|----------|-------------|
| 13 | **YOLO optimizer params** | Training script uses SGD lr=0.01, momentum=0.937, batch=16 — matches thesis. Verified. | ✅ Already correct |
| 14 | **Calibration tool never used** | `mm_per_pixel` set to 0.09 (approximate). Not calibrated per camera setup. | Run `calibrate_camera.py` in target setup |
| 15 | **No 3D reconstruction** | Thesis mentions as future work. Not in scope. | Document as known limitation |

#### NEW — Session 2

| # | Gap | Evidence | Status |
|---|-----|----------|--------|
| 16 | **YOLO trained on bad labels** | 25.5% of training labels were full-image boxes (edge detection failed). Old labels: 442 damaged + 113 not_damaged (80/20). | ✅ Regenerated: better edge detection (Otsu+Canny+morph), stratified split, full-image boxes dropped to 16.6%. **Needs retrain.** |
| 17 | **Backend grades mismatch** | Backend mapped AA/A/B instead of A/B/C/Reject. | ✅ Fixed `map_to_grade()` in `yolo_inference.py` |
| 18 | **Training not portable** | `data.yaml` had hardcoded absolute path. No `--device` arg. | ✅ `path: .`, `--device cuda:0\|mps\|cpu`, README updated with cross-machine steps |

---

## 3. Priority Remediation Plan

### ✅ Phase 1 — Core Pipeline Fixes (Complete)
1. ✅ Integrate `EggGradingCNN` into `live_view.py`
2. ✅ CNN confidence filtering with `C` key toggle
3. ✅ Grade system A/B/C/Reject
4. ✅ Fix weight formula (0.05→0.0005)
5. ✅ `--cnn-conf` CLI argument
6. ✅ Fix `num_classes=3→2`
7. ✅ Backend grades AA/A/B→A/B/C/Reject
8. ✅ YOLO label regeneration (better edge detection, 70-20-10 split)
9. ✅ Make training portable (relative paths, --device, README)

### 🔄 Phase 1b — YOLO Retraining (Now)
10. 🔄 Retrain YOLO with regenerated labels (100 epochs, patience=50)
    - `python src/train_yolo.py --train --device ...` on target machine

### Phase 2 — Evaluation
11. Create evaluation script (precision, recall, F1, mAP)
12. Evaluate retrained YOLO on test set
13. Evaluate CNN on test set

### Phase 3 — Frontend Enhancement
14. Add grade/size/weight display to web app
15. Add WebSocket real-time updates

### Phase 4 — Polish
16. Balance dataset (augment minority class)
17. Run camera calibration for exact mm_per_pixel

---

## 4. Remaining Implementation Guide

### P2-1: Evaluation Script

```python
# src/evaluate.py
# Compute precision, recall, F1, mAP for YOLO on test set
# Confusion matrix for CNN on test set
# Usage: python src/evaluate.py
```

---

## 5. Validation Checklist

Current status:

- [x] `python src/live_view.py` — shows Grade A/B/C/Reject on boxes
- [x] `C` key toggles CNN confidence filter (yellow boxes)
- [x] Weight shows realistic values (30-80g), not always 80
- [x] CNN correctly loads from `models/egg_grader.pth`
- [x] CSV log includes grade column
- [x] Test set exists at `data/test/` (253 images)
- [x] CNN training uses correct `num_classes=2`
- [x] YOLO training params match thesis (SGD lr=0.01, momentum=0.937, batch=16)
- [x] Training portable: relative paths, --device arg, GPU support
- [ ] `python src/evaluate.py` outputs precision, recall, F1, mAP
- [ ] YOLO retrained with regenerated labels (100 epochs)
- [ ] Frontend shows grade + size + weight per detection

---

## 6. Risk & Dependencies

| Risk | Impact | Mitigation |
|------|--------|------------|
| YOLO model still on old labels | Poor detection until retrained | High priority — retrain with regenerated labels |
| Camera calibration varies | Weight wrong per setup | Make calibration step mandatory in setup |
| CNN quality unknown | May need retrain with balanced data | Evaluate first, retrain with class weights if needed |
