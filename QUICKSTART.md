# Quick Start Guide - Enhanced Egg Detection System

## Installation

1. **Install dependencies** (if not already installed):
```bash
pip install ultralytics opencv-python torch pyyaml
```

2. **Verify files exist**:
```bash
ls -l models/egg_grader.pth
ls -l runs/detect/egg_detection/train1/weights/best.pt
ls -l config/config.yaml
```

## Step 1: Camera Calibration (Optional but Recommended)

Before running detection, calibrate your camera for accurate size/weight measurement:

```bash
python3 src/calibrate_camera.py --source 0
```

**Calibration Steps**:
1. Place a reference object (ruler, known-size card, or standard egg) in front of camera
2. Press **R** to start measurement
3. Draw a rectangle around the reference object
4. Enter the actual size in millimeters
5. Press **S** to save

This creates the `mm_per_pixel` value in `config/config.yaml`.

## Step 2: Run Static Tray Detection

Start the FastAPI backend and upload one top-down tray image (0–5 eggs) through
`POST /api/v1/predictions/upload`:

```bash
cd backend
uvicorn app.main:app --reload
```

## Step 3: View Results

After running detection, check:

1. **API response**:
   - Number of eggs detected
   - Size and weight per egg
   - Damage classification and grade

2. **CSV Log File** (`egg_statistics.csv`):
```bash
# View all detections
cat egg_statistics.csv

# Analyze with Python
python3 -c "
import pandas as pd
df = pd.read_csv('egg_statistics.csv')
print('Total eggs detected:', len(df))
print('Damaged eggs:', len(df[df['class'] == 'Damaged']))
print('Average weight:', df['weight_g'].mean())
print('Weight by size:')
print(df.groupby('size_category')['weight_g'].mean())
"
```

## Configuration Options

Edit `config/config.yaml`:

```yaml
detection:
  yolo_confidence: 0.75
  cnn_confidence: 0.70

calibration:
  mm_per_pixel: 0.09
```

## Troubleshooting

### "Could not load YOLO model"
- Check: `ls runs/detect/egg_detection/train1/weights/best.pt`
- Solution: Verify YOLO training completed successfully

### Weight always same value
- Run calibration: `python3 src/calibrate_camera.py`
- Check `config.yaml` `mm_per_pixel` is not 1.0

## Analyze Results

```bash
# After grading images through the API:
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('egg_statistics.csv')
print(f"Total eggs: {len(df)}")
print(f"Damaged: {(df['class'] == 'Damaged').sum()}")
print(f"Average weight: {df['weight_g'].mean():.1f}g")
print(f"\nSize breakdown:")
for size in ['Small', 'Medium', 'Large']:
    count = (df['size_category'] == size).sum()
    avg_weight = df[df['size_category'] == size]['weight_g'].mean()
    print(f"  {size}: {count} eggs, avg {avg_weight:.1f}g")
EOF
```

## What's New

✅ **Reduced false positives** - Higher default confidence threshold
✅ **Better classification** - CNN confidence scores with filtering
✅ **Static tray processing** - Supports one image with 0–5 eggs
✅ **Real-world measurements** - Camera calibration for mm/weight
✅ **Data logging** - All detections saved to CSV

## Next Steps

1. Calibrate your camera for accurate measurements
2. Start the FastAPI backend
3. Upload top-down tray images to `/api/v1/predictions/upload`
4. Review the returned results and `egg_statistics.csv`

See `IMPROVEMENTS.md` for detailed documentation on all features.
