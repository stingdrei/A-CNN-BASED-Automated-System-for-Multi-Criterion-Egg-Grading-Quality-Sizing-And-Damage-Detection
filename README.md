# CNN-Based Automated System for Multi-Criterion Egg Grading

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is a starter template for an automated egg grading system using Convolutional Neural Networks (CNNs), YOLO object detection, and OpenCV. It targets multi-criterion grading: quality, sizing, and damage detection.

> **Current beta status:** cleanliness is not implemented, so final grades are
> `N/A`. The API exposes `provisional_grade` from damage and size only. Physical
> size/weight calibration and the 0-egg/0-5 tray camera gate are not validated.

## Features
- Modular data loading and image preprocessing (OpenCV)
- CNN architecture for damage classification (PyTorch)
- YOLOv8 object detection for egg localization
- Static tray image grading through the FastAPI backend
- Camera calibration hooks for size/weight measurements (not yet validated)
- Per-egg CSV export with canonical grading fields
- Provisional grading while cleanliness is unavailable
- Clearly separated training and inference scripts
- Easy expansion for new grading criteria

## Project Structure

```
.
├── backend/            FastAPI app, SQLAlchemy models, Alembic migrations
├── frontend/           React + TypeScript + Tailwind dashboard
├── src/                training, inference, evaluation, calibration scripts
├── tests/              backend/API/model/rule tests
├── egg_detection/      YOLO checkpoints and run artifacts
├── data/               detector/damage datasets and manifests
├── models/             damage checkpoints, model docs, checksums
├── reports/            locked test reports, CSV/JSON evidence
├── config/             training/calibration example config
├── TASKS/              implementation and release-gate checklists
└── requirements.txt
```

### Folder Descriptions
- `backend/`: authenticated API, persistence, inference orchestration.
- `frontend/`: upload/camera, result, history, and dashboard UI.
- `src/`: reusable model/training/evaluation code.
- `data/`: source data, generated crops, detector datasets, manifests.
- `models/`: model cards, checkpoints, and checksums.
- `reports/`: reproducible evaluation evidence.
- `TASKS/`: task status and release gates.

## Setup
1. Clone repository & install dependencies:
    ```bash
    pip install -r requirements.txt
    pip install -r backend/requirements.txt
    ```
2. Place your dataset in the `data/` folder.
3. Apply database migrations from `backend/`:
    ```bash
    cd backend
    alembic upgrade head
    ```
4. (Optional) Calibrate your camera for accurate measurements:
    ```bash
    python src/calibrate_camera.py --source 0
    ```
5. Start the FastAPI backend:
    ```bash
    ./.venv/bin/uvicorn app.main:app --app-dir backend --reload
    ```

Use `./.venv/bin/python -m pip` when installing packages: on this workstation
`.venv/bin/pip` can point at a different interpreter. OpenCV 4.9 requires
`numpy<2`.

### Production Configuration and Security

Copy `.env.example` to `.env` and set the deployment values. Never commit
`.env`.

```bash
cp .env.example .env
openssl rand -hex 32   # use this value for SECRET_KEY
```

Required before `ENVIRONMENT=production` will start:

- unique `SECRET_KEY` of at least 32 characters;
- `DEBUG=false`;
- explicit `CORS_ORIGINS`, never `*`;
- non-local `DATABASE_URL`.

Example:

```bash
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql://eggs:password@db.example/eggs
SECRET_KEY=<64-hex-character-value>
CORS_ORIGINS=https://eggs.example
```

Run behind HTTPS. The backend rejects video uploads and only accepts static
images (`jpg`, `jpeg`, `png`, `webp`, `bmp`) with a configurable size limit.

## Usage

### YOLO Training (Local)
```bash
# Prepare dataset directories & data.yaml
python src/train_yolo.py --prepare

# Audit image/label pairing and split leakage before training
python src/normalize_detector_labels.py --in-place
python src/verify_detection_dataset.py \
  --preview-dir /tmp/egg-box-previews \
  --preview-per-split 20

# Train (CPU — ~2.5hr for 100 epochs on M1)
python src/train_yolo.py --train

# After selecting the confidence threshold on validation, evaluate once on test
python src/evaluate_detector.py \
    --model egg_detection/train1/weights/best.pt \
    --confidence 0.75

# Train on GPU
python src/train_yolo.py --train --device cuda:0
python src/train_yolo.py --train --device mps       # Apple Silicon
```

### Safe Fine-Tuning From the Current Detector

Create a separate one-class dataset and fine-tune from the current detector
without modifying its images, labels, or checkpoint:

```bash
python src/prepare_detector_finetune.py \
    --source-yaml data/detection/data.yaml \
    --output-root data/detection_finetune

python src/verify_detection_dataset.py \
    --data-yaml data/detection_finetune/data.yaml \
    --annotation-source "manually verified; fine-tuning dataset" \
    --manifest data/detection_finetune/manifest.csv \
    --report reports/detection_finetune_dataset_report.json

python src/train_yolo.py --train \
    --model egg_detection/train1/weights/best.pt \
    --data-root data/detection_finetune \
    --project egg_detection \
    --name finetune_egg_v1 \
    --device mps \
    --epochs 50
```

The fine-tuning run writes to `egg_detection/finetune_egg_v1/`; it does not
overwrite `egg_detection/train1/weights/best.pt`. Fine-tuning starts from a
copy of the learned weights, so it retains prior egg-localization knowledge,
but new training can still reduce performance when labels are wrong or the
dataset is narrow. Keep the original checkpoint and compare both models on
the same locked test set before replacing it.

The committed run reports validation metrics of mAP50=0.96576 and
precision=0.96487 at epoch 50. On the shared 19-image/57-egg test split at
confidence 0.75, the fine-tuned checkpoint achieved precision=0.9584,
recall=1.0000, mAP50=0.9860, mAP50-95=0.9135, exact tray-count accuracy
0.8947, 0 missed eggs per tray, and 0.1579 false positives per tray. The
baseline scored precision=0.8696, recall=0.3509, mAP50=0.5948,
mAP50-95=0.4630, exact tray-count accuracy 0.6316, 0.3158 missed eggs per
tray, and 0.3158 false positives per tray. The fine-tuned checkpoint is now
the backend and local-camera default; the baseline remains available for
regression comparisons.

To reproduce the fine-tuned test evaluation:

```bash
python src/evaluate_detector.py \
    --model egg_detection/finetune_egg_v1/weights/best.pt \
    --data-yaml data/detection_finetune/data.yaml \
    --confidence 0.75 \
    --output reports/detector_finetune_test_report.json
```

The confidence threshold was selected on validation data before this test
evaluation. The complete reports are stored in
`reports/detector_baseline_finetune_split_report.json` and
`reports/detector_finetune_test_report.json`.

### Damage Data v2 and Crop Evaluation

The old damage manifests were incomplete. Rebuild them from every supported
image, remove exact cross-label duplicates, and verify file/hash integrity:

```bash
python src/create_damage_split.py \
    --source data/damage \
    --output data/damage/manifests_v2 \
    --report reports/damage_dataset_v2_report.json

python src/verify_damage_dataset.py \
    --manifest-dir data/damage/manifests_v2 \
    --image-root . \
    --report reports/damage_dataset_v2_verification.json
```

Current v2 result: 1163 images found, 4 exact duplicate files (2 conflicting
label groups) removed, 1159 used; split 811/174/174. The two conflicting pairs
are excluded from the manifests and recorded in the report. Strict
source/session isolation is **not** possible: the repository exposes three
source groups and all 369 broken-egg images came from one capture session.

Build detector-cropped damage evaluation images and evaluate the current
crop model on the same pipeline crops:

```bash
python src/build_damage_crops.py \
    --manifest data/damage/manifests_v2/test.csv \
    --output data/damage_crops/test \
    --report reports/damage_crop_test_report.json \
    --model egg_detection/finetune_egg_v1/weights/best.pt \
    --confidence 0.75 \
    --clean

python src/evaluate_classifier.py \
    --labels data/damage_crops/test/manifest.csv \
    --image-root . \
    --checkpoint models/egg_grader_crop_resnet18_v1.pth \
    --output reports/damage_crop_classifier_resnet18_v1_test_report.json
```

Current crop-set result: 160/174 images detected, 28 multi-detection images
skipped as label-ambiguous, 14 missed, 132 crops used (84 damaged, 48 not
damaged). The legacy full-image checkpoint scores 0.394 accuracy on these
crops; the custom crop fine-tune scores 0.879; the ResNet18 crop model scores
0.985 accuracy, 0.979 balanced accuracy, and 1.000 damaged recall at the
validation-selected threshold 0.18. The backend defaults to the ResNet18
checkpoint. The test set was inspected during threshold exploration, so this
is provisional evidence and T7 remains closed.

### Detector Smoke Subset (1-5 eggs)

```bash
python src/build_detector_smoke.py \
    --data-yaml data/detection_finetune/data.yaml \
    --output data/detection_smoke \
    --report reports/detector_smoke_subset_report.json \
    --min-eggs 1 --max-eggs 5 --clean

python src/evaluate_detector.py \
    --model egg_detection/finetune_egg_v1/weights/best.pt \
    --data-yaml data/detection_smoke/data.yaml \
    --split test \
    --confidence 0.75 \
    --output reports/detector_smoke_test_report.json
```

The smoke subset has 17 test images/41 eggs: exact tray-count accuracy 0.9412,
0 missed eggs per tray, 0.0588 false positives per tray, mAP50 0.9881. This is
a demo check only; it is not the full 0-5 tray beta gate.

Tray post-processing removes duplicate boxes at IoU 0.85. Counts `1–5` are
`ok`, `0` is `empty`, and more than `5` is `over_capacity`: all detections are
kept, `tray_count_valid=false`, and every egg is marked `requires_review` with
`tray_over_capacity`. The count is never silently capped.

Release evidence still needs physical 0-egg/0–5 trays. The current locked test
has no 0-egg image and includes two trays with more than 5 eggs.

### Training on Another Machine
Transfer the 307MB dataset to any machine with Python:

```bash
# 1. Copy dataset
rsync -avz user@this-machine:/path/egg-cv/data/detection/ /path/to/egg-cv/data/detection/

# 2. Copy the repo (or just train_yolo.py + data/detection/)
# 3. Install dependencies
pip install ultralytics pyyaml

# 4. Train (adjust device to match hardware)
python src/train_yolo.py --train --device cuda:0    # NVIDIA GPU
python src/train_yolo.py --train --device mps       # Apple Silicon GPU
python src/train_yolo.py --train                    # CPU fallback

# 5. Copy back the trained weights
rsync -avz /path/to/egg-cv/egg_detection/train1/weights/best.pt \
    user@this-machine:/path/egg-cv/models/
```

The detector uses one class (`egg`); damage is classified downstream. The
normalization command converts legacy damage-status class IDs to class 0.
For the fine-tune dataset, the audit writes
`data/detection_finetune/manifest.csv` and
`reports/detection_finetune_dataset_report.json` when those flags are supplied.
Training should not begin if the audit reports duplicate or near-duplicate
leakage, invalid labels, missing image/label pairs, or unverified annotation
problems. The current generated
labels are heuristic contour boxes and require visual verification before
detector metrics are considered valid. With `--preview-dir`, inspect the
annotated samples under `/tmp/egg-box-previews/{train,val,test}`. The report
also lists boxes covering at least 75% of an image under `oversized_boxes`;
these require particular attention because they may indicate classification
images rather than tightly annotated egg locations.

The `data/detection/data.yaml` uses repository-relative paths and
`train_yolo.py` resolves them independently of the caller's working directory.

### Static Tray Detection
```bash
# Start the backend, then upload a tray image to POST /api/v1/predictions/upload
./.venv/bin/uvicorn app.main:app --app-dir backend --reload
```

The compatibility routes `POST /api/v1/grade`, `GET /api/v1/records`, and
`GET /api/v1/stats` map to the same authenticated operations.

### Browser Camera Capture

The upload page supports a one-shot camera mode at
`/upload?mode=camera`. It requests the rear camera, shows a live preview, and
posts one JPEG frame when **Capture & Analyze** is pressed. The result page has
a **Next Tray** button that returns to the camera. Camera access requires HTTPS
or `localhost`.

The browser-camera MVP is implemented but not release-approved: the physical
0-egg/0-5 tray checks and the cleanliness/size/weight gates remain open.

### Weight Calibration

Fit `W = k × L × B²` from weighed eggs after collecting a CSV with
`length_mm`, `breadth_mm`, and `actual_weight_g`:

```bash
python src/calibrate_weight.py \
    --data data/weight_calibration.csv \
    --output reports/weight_calibration_report.json
```

The report includes `k`, a 95% interval, R², MAE, RMSE, signed error, and the
±3 g rate. The backend keeps `WEIGHT_K=0.0` until a fit is accepted; set
`WEIGHT_K` and `CALIBRATION_VERSION` to activate it.

### Provisional Grading Contract

Every per-egg result now carries:

- `grade`: final grade; `N/A` while cleanliness is unavailable.
- `provisional_grade`: damage+size decision (`A/B/C/Reject/N/A`); not final.
- `size`: canonical `Small/Medium/Large/Unknown`.
- `damage_status`, `cleanliness_status`.
- `confidence`: nested `detection`, `damage`, `cleanliness`.
  `cleanliness_status=Unavailable` until a cleanliness checkpoint is present.
- `geometry`: nested `length_mm`, `breadth_mm`, `quality`.
- `requires_review`, `review_reason`.
- `model_versions`: nested detector/damage/cleanliness/calibration versions.

The API, database, CSV, and frontend use these names. Final A/B/C requires
`damage_status == "Not Damaged"` and `cleanliness_status == "Clean"`.
`provisional_grade` deliberately ignores cleanliness, so it must be shown as
provisional in UI and reports.

The per-egg CSV is appended to `reports/egg_statistics_v2.csv` by default.
Set `EGG_CSV_PATH` to override it. The schema is defined in
`backend/app/services/csv_service.py`.

### Live Camera Detector (No Frontend)
```bash
./.venv/bin/python src/live_camera.py --device mps
```

The camera window shows YOLO detections and the current egg count. Press `q` or
`Esc` to stop. Use `--camera 1` if the default camera is unavailable. This is a
detector-only demonstration; it does not save predictions or run the API.

### Analyze Results
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('reports/egg_statistics_v2.csv')
print('Total eggs:', len(df))
print('Damaged:', (df['damage_status'] == 'Damaged').sum())
print('Provisional A:', (df['provisional_grade'] == 'A').sum())
print('Final N/A:', (df['grade'] == 'N/A').sum())
print('Avg weight:', df['weight_g'].mean())
"
```

`weight_g` is empty until `WEIGHT_K` is fitted from weighed eggs.

## Requirements
- Python 3.9+
- PyTorch ≥ 2.0
- OpenCV ≥ 4.8
- Ultralytics (YOLOv8)
- (see requirements.txt)

## Configuration

Runtime settings come from environment variables / `.env` and are defined in
`backend/app/config.py`. See `.env.example` for the full list, including:

- `CONFIDENCE_THRESHOLD` — YOLO detections (default: 0.75)
- `DAMAGE_DECISION_THRESHOLD` — damage classifier (default: 0.18)
- `MAX_TRAY_EGGS` — static-tray capacity (default: 5)
- `DETECTION_DUPLICATE_IOU` — duplicate suppression (default: 0.85)
- `MM_PER_PIXEL`, `WEIGHT_K`, `CALIBRATION_VERSION` — measurement calibration

`config/config.yaml` is retained for training/calibration scripts and is not the
runtime API configuration.

Checkpoint SHA-256 values are recorded in `models/CHECKSUMS.md`.

## Extend
- Add new grading criteria to `src/dataset.py` and `src/model.py`
- Retrain YOLO with custom annotations via `src/train_yolo.py`
- Integrate new data sources in the FastAPI prediction service

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
MIT
