# CNN-Based Automated System for Multi-Criterion Egg Grading

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is a starter template for an automated egg grading system using Convolutional Neural Networks (CNNs), YOLO object detection, and OpenCV. It targets multi-criterion grading: quality, sizing, and damage detection.

## Features
- Modular data loading and image preprocessing (OpenCV)
- CNN architecture for damage classification (PyTorch)
- YOLOv8 object detection for egg localization
- Static tray image grading through the FastAPI backend
- Camera calibration for accurate size/weight measurements
- CSV data logging for all detections
- Clearly separated training and inference scripts
- Easy expansion for new grading criteria

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train_yolo.py
│   ├── convert_to_yolo.py
│   ├── calibrate_camera.py
│   └── utils.py
├── models/
├── notebooks/
└── runs/
```

### Folder Descriptions
- `data/`: Place egg images here; split as needed (raw/processed).
- `models/`: Saved models and checkpoints.
- `notebooks/`: For development and analysis.
- `src/`: Core source code.
- `config/`: Configurations (YAML, JSON, etc.).
- `runs/`: YOLO training outputs and detection runs.

## Setup
1. Clone repository & install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Place your dataset in the `data/` folder.
3. (Optional) Calibrate your camera for accurate measurements:
    ```bash
    python src/calibrate_camera.py --source 0
    ```
4. Start the FastAPI backend:
    ```bash
    cd backend
    uvicorn app.main:app --reload
    ```

## Usage

### YOLO Training (Local)
```bash
# Prepare dataset directories & data.yaml
python src/train_yolo.py --prepare

# Train (CPU — ~2.5hr for 100 epochs on M1)
python src/train_yolo.py --train

# Train on GPU
python src/train_yolo.py --train --device cuda:0
python src/train_yolo.py --train --device mps       # Apple Silicon
```

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

The `data/detection/data.yaml` uses relative paths so it works anywhere.

### Static Tray Detection
```bash
# Start the backend, then upload a tray image to POST /api/v1/predictions/upload
cd backend
uvicorn app.main:app --reload
```

### Analyze Results
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('egg_statistics.csv')
print('Total eggs:', len(df))
print('Damaged:', (df['class'] == 'Damaged').sum())
print('Avg weight:', df['weight_g'].mean())
"
```

## Requirements
- Python 3.8+
- PyTorch ≥ 2.0
- OpenCV ≥ 4.8
- Ultralytics (YOLOv8)
- (see requirements.txt)

## Configuration

Edit `config/config.yaml` to adjust:
- `detection.yolo_confidence` — YOLO detection threshold (default: 0.75)
- `detection.cnn_confidence` — CNN classification threshold (default: 0.70)
- `calibration.mm_per_pixel` — Camera calibration factor

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
