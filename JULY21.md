# Thesis Gap Analysis & Remediation Plan

## Date: July 21, 2026

## Source Documents
- **THESIS.pdf** — Academic thesis specification (unreadable by LLM, reconstructed from `THESIS_DESIGN.md` and `THESIS_METHODOLOGY_DESIGN.md`)
- **THESIS_DESIGN.md** — System design spec
- **THESIS_METHODOLOGY_DESIGN.md** — Training methodology spec
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
| YOLOv8s detection | ✅ Fine-tuned | `models/egg_detection_finetuned/weights/best.pt` |
| EggGradingCNN architecture | ✅ 3→32→64, 2 FC | `src/model.py` |
| Weight formula W=0.05×D³ | ✅ Implemented | `src/live_view.py:92`, `backend/app/ml/yolo_inference.py:51` |
| Size categories | ✅ S<50, M 50-60, L>60 | `src/live_view.py:101-106` |
| Weight clamp 30-80g | ✅ | `src/live_view.py:93` |
| Centroid tracking | ✅ ObjectTracker class | `src/live_view.py:10-79` |
| CSV logging | ✅ StatisticsLogger class | `src/live_view.py:109-151` |
| Camera calibration | ✅ CameraCalibrator class | `src/live_view.py:82-106` |
| Calibration tool | ✅ Interactive script | `src/calibrate_camera.py` |
| Data augmentation | ✅ Horizontal flips, 224×224, ImageNet norm | `src/utils.py` (transforms) |
| YOLO confidence threshold | ✅ Default 0.75, keyboard +/- | `src/live_view.py:291-296` |
| CNN training params | ✅ Adam lr=0.001, 20 epochs | `src/train.py:34` |
| Color-coded boxes | ✅ Green/Red | `src/live_view.py:268-273` |
| Backend API | ✅ FastAPI + PostgreSQL | `backend/` |
| Frontend dashboard | ✅ React + Vite | `frontend/` |

### Gaps (Not Matching Thesis)

#### CRITICAL

| # | Gap | Evidence | Fix Required |
|---|-----|----------|-------------|
| 1 | **CNN not loaded in live_view.py** | `live_view.py` imports YOLO only (line 155). No `EggGradingCNN` import. Bypasses CNN classifier entirely. YOLO outputs used directly for damage classification. | Load `models/egg_grader.pth` → crop detections to 224×224 → run CNN → use CNN output for damage class |
| 2 | **Weight always 80g** | `mm_per_pixel=1.0` (default). Egg bboxes are 720-1080px wide. Real eggs ~50-60mm. With 1.0 mm/px, diameter_mm=720-1080 → weight clamped to 80g every time. | Run calibration or set realistic mm_per_pixel (~0.08-0.1 for typical webcam setups) |
| 3 | **No grade system in live_view** | `live_view.py:267-273` shows class + confidence only. No Grade A/B/C/Reject mapping. Backend `yolo_inference.py:54-64` has `map_to_grade()` but live_view doesn't use it. | Add grade mapping and display |

#### HIGH

| # | Gap | Evidence | Fix Required |
|---|-----|----------|-------------|
| 4 | **No CNN confidence filtering** | Thesis: CNN confidence threshold 0.60-0.70 with toggle. Code has YOLO threshold only. `C` key toggle referenced in docs but not in code. | Add `predict_image()` returning softmax confidence, `C` key toggle, yellow boxes for low confidence |
| 5 | **Train/val/test split wrong** | Thesis: 70-10-10. Code: 80-20 (no held-out test set). | Create `split_train_val_test.py` with 70-20-10 stratified split |
| 6 | **No evaluation harness** | Thesis: precision, recall, F1, mAP required. No script computes these. | Create `evaluate.py` — confusion matrix, precision, recall, F1, mAP@50 |
| 7 | **Frontend lacks grading** | Thesis requires JSON with grade/size/weight. Frontend shows detections but no grade info. | Add grade, size, weight to result display |

#### MEDIUM

| # | Gap | Evidence | Fix Required |
|---|-----|----------|-------------|
| 8 | **Dataset class imbalance** | 632 Damaged : 162 Not Damaged (80:20 ratio). CNN may bias toward Damaged. | Augment Not Damaged class, add class weights to loss |
| 9 | **CNN output classes mismatch** | `model.py:6` defaults `num_classes=3`. Config says 2. Original train may have used 3 classes. | Fix to `num_classes=2`, retrain |
| 10 | **No WebSocket real-time** | Backend has WebSocket stub in `MVP_ARCHITECTURE.md` but no implementation. | Connect WebSocket for live updates |
| 11 | **Frontend no size/weight** | Thesis requires size category + weight in output. Frontend doesn't show these. | Add size/weight to Result page |

#### LOW

| # | Gap | Evidence | Fix Required |
|---|-----|----------|-------------|
| 12 | **YOLO trained 50 epochs not 20** | Thesis: 20 epochs SGD. Code: 50 epochs. Impact unknown. | Retrain 20 epochs per spec or justify difference |
| 13 | **YOLO optimizer params unverified** | Thesis: SGD lr=0.01, momentum=0.937, batch=16. YOLO training script may use ultralytics defaults. | Verify/align training params |
| 14 | **Calibration tool never used** | `mm_per_pixel` still 1.0. No calibration performed. | Run `calibrate_camera.py` in target setup |
| 15 | **No 3D reconstruction** | Thesis mentions as future work. Not in scope. | Document as known limitation |

---

## 3. Priority Remediation Plan

### Phase 1 — Core Pipeline Fixes (Must-Have)
1. Integrate `EggGradingCNN` into `live_view.py`
2. Implement CNN confidence filtering with `C` key toggle
3. Add grade system (A/B/C/Reject)
4. Fix weight estimation (calibrate mm_per_pixel)
5. Add `--cnn-conf` CLI argument

### Phase 2 — Evaluation
6. Create evaluation script (precision, recall, F1, mAP)
7. Create 70-20-10 data split
8. Re-train CNN with correct `num_classes=2`

### Phase 3 — Frontend Enhancement
9. Add grade/size/weight display to web app
10. Add WebSocket real-time updates

### Phase 4 — Polish
11. Balance dataset (augment minority class)
12. Align YOLO training params with thesis
13. Document all deviations from thesis

---

## 4. Detailed Implementation Guide

### P1-1: Integrate EggGradingCNN into live_view.py

```python
# Add to imports in src/live_view.py:
import torch
import torch.nn.functional as F
from model import EggGradingCNN
from torchvision import transforms

# Add after YOLO model load:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = EggGradingCNN(num_classes=2)
cnn_model.load_state_dict(torch.load('models/egg_grader.pth', map_location=device))
cnn_model.to(device)
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

```python
# Replace YOLO class output in the detection loop:
# After cropping egg from frame:
egg_crop = frame[y1:y2, x1:x2]
if egg_crop.size > 0:
    input_tensor = cnn_transform(egg_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        output = cnn_model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred = torch.argmax(output, 1).item()
        cnn_conf = float(probs[pred].cpu().numpy())
    
    class_names_cnn = ["Not Damaged", "Damaged"]
    cnn_label = class_names_cnn[pred] if pred < len(class_names_cnn) else "unknown"
else:
    cnn_label = "unknown"
    cnn_conf = 0.0
```

### P1-2: CNN Confidence Filtering

```python
# Add CLI arg:
parser.add_argument("--cnn-conf", type=float, default=0.6,
                    help="CNN confidence threshold (default 0.6)")

# Add toggle state:
enable_cnn_filtering = True

# In tracking loop after CNN classification:
if enable_cnn_filtering and cnn_conf < args.cnn_conf:
    display_class = "Low Confidence"
    box_color = (0, 255, 255)  # Yellow
else:
    display_class = cnn_label
    box_color = (0, 255, 0) if cnn_label == "Not Damaged" else (0, 0, 255)

# C key handler:
elif key == ord('c'):
    enable_cnn_filtering = not enable_cnn_filtering
    print(f"CNN Filter: {'ON' if enable_cnn_filtering else 'OFF'}")
```

### P1-3: Grade System

```python
def map_grade(damage_status, size_category):
    if damage_status == "Damaged":
        return "Reject"
    if size_category == "Large":
        return "A"
    elif size_category == "Medium":
        return "B"
    elif size_category == "Small":
        return "C"
    return "N/A"

# Per-detection in loop:
grade = map_grade(display_class, size)
# Display: "Grade A | 68.5g"
```

### P1-4: Fix Weight

```yaml
# config/config.yaml — update calibration:
calibration:
  mm_per_pixel: 0.09  # Approximate for 1080p at 15cm distance
  # Run calibrate_camera.py to get exact value
```

Or use known reference: typical egg diameter ~45mm. If bbox width is ~500px: mm_per_pixel = 45/500 = 0.09.

### P2-1: Evaluation Script

```python
# src/evaluate.py
# Compute precision, recall, F1, mAP for YOLO on test set
# Confusion matrix for CNN on test set
# Usage: python src/evaluate.py
```

### P2-2: Data Split

```
data/
  eggs/
    images/       (all images)
    labels/       (YOLO labels)
    data.yaml
  train/          (70%)
  val/            (10%)
  test/           (20%)
```

---

## 5. Validation Checklist

After all phases complete, verify:

- [ ] `python src/live_view.py` — shows Grade A/B/C/Reject on boxes
- [ ] `C` key toggles CNN confidence filter
- [ ] Weight shows realistic values (30-80g), not always 80
- [ ] `python src/evaluate.py` outputs precision, recall, F1, mAP
- [ ] CNN correctly loads from `models/egg_grader.pth`
- [ ] Frontend shows grade + size + weight per detection
- [ ] CSV log includes grade column
- [ ] Test set exists at `data/test/`
- [ ] YOLO training params match thesis (20 epochs, SGD)
- [ ] CNN training uses correct `num_classes=2`

---

## 6. Risk & Dependencies

| Risk | Impact | Mitigation |
|------|--------|------------|
| `egg_grader.pth` trained with `num_classes=3` | Retrain needed | Check model weight shape; retrain if mismatch |
| No labeled test set | Can't eval | Create 70-20-10 split before eval |
| Camera calibration varies | Weight wrong per setup | Make calibration step mandatory in setup |
| CNN quality unknown | May need retrain with balanced data | Evaluate first, retrain with class weights if needed |
