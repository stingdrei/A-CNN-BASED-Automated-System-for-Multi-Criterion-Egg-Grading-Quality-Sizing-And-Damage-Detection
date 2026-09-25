# Enhanced Egg Detection System - Architecture & Data Flow

## Current Target Data Flow

The detector and classifiers have separate responsibilities. YOLO must detect
only eggs; the damage and cleanliness models classify each detected egg crop.
The grading engine combines those results with calibrated geometry.

```mermaid
flowchart TD
    A[Camera or static tray image] --> B[YOLO egg detector<br/>class: egg]
    B --> C{Confidence and bounds validation}
    C -->|valid detection| D[Crop each egg<br/>with padding and clipping]
    C -->|invalid or low confidence| U[Unknown / surfaced failure]

    D --> E[Damage classifier<br/>Damaged / Not Damaged]
    D --> F[Cleanliness classifier<br/>Clean / Stained]
    D --> G[Geometry measurement<br/>major axis L, minor axis B]

    G --> H[Camera calibration<br/>pixels to millimeters]
    H --> I[Size and weight estimator<br/>size bins + W = k × L × B²]

    E --> J[Per-egg result record]
    F --> J
    I --> J
    B --> J
    J --> K{Deterministic grading rules}
    K -->|Damaged or Stained| L[Reject]
    K -->|Clean + Large| M[Grade A]
    K -->|Clean + Medium| N[Grade B]
    K -->|Clean + Small| O[Grade C]
    K -->|Unknown or invalid| P[N/A or abstain]

    J --> Q[API response / CSV / database]
    L --> Q
    M --> Q
    N --> Q
    O --> Q
    P --> Q
    Q --> R[Dashboard, annotated image, audit metadata]
```

### Model connection contract

| Stage | Input | Output | Next stage |
|---|---|---|---|
| YOLO egg detector | Full tray frame | One `egg` box and detection confidence per egg | Cropper |
| Cropper | Frame plus box | One bounded egg crop per detection | Damage, cleanliness, geometry |
| Damage classifier | Egg crop | `damage_status`, damage confidence | Grading |
| Cleanliness classifier | Egg crop | `cleanliness_status`, cleanliness confidence | Grading |
| Geometry/calibration | Egg crop and camera calibration | `L`, `B`, millimeters, size, weight | Grading |
| Grading engine | All per-egg outputs | `grade` (`A`, `B`, `C`, `Reject`, or `N/A`) | API/storage/UI |

The detector should not use `damaged` and `not_damaged` as its object classes.
Those are classifier outputs. This separation allows one tray image to produce
up to five independent egg results and prevents non-egg objects from being
treated as damaged or undamaged eggs.

## Legacy Prototype Flow

The ASCII diagram below describes the earlier tracking prototype and is retained
for historical reference only. It is not the current static-tray target
architecture: do not add video tracking, FPS requirements, or tracking IDs to
the final implementation.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT (Camera/File)                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   YOLO OBJECT DETECTION                         │
│                  (YOLOv8 Small Model)                           │
│  Input: Full frame  │  Output: Bounding boxes + confidence      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONFIDENCE FILTERING (Threshold: 0.65)             │
│            Filter detections below YOLO confidence             │
│              (Reduces false positives by ~70%)                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OBJECT TRACKING                                │
│                (Centroid-Based Tracking)                        │
│                                                                 │
│  Matches detections across frames:                             │
│  - Compute centroids of new detections                         │
│  - Match to existing tracked objects (if distance < threshold) │
│  - Assign unique IDs to new eggs                               │
│  - Count only on first appearance (is_new=True)                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
      ┌───────────────┴────────────────┐
      │                                │
      ▼                                ▼
┌──────────────────┐          ┌──────────────────┐
│ Crop egg region  │          │ Update tracking  │
│ from frame       │          │ state            │
└────────┬─────────┘          └──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              CNN DAMAGE CLASSIFICATION                          │
│            (EggGradingCNN - 2 layer network)                    │
│                                                                 │
│  Input: Cropped egg (224×224)                                 │
│  Output: Class (Damaged/Not Damaged) + Confidence score       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│         CNN CONFIDENCE FILTERING (Threshold: 0.6)              │
│     Only accept classifications with high confidence           │
│        (Toggle on/off with 'C' key during runtime)            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              CAMERA CALIBRATION (Real-world Conversion)        │
│                                                                 │
│  Diameter (pixels) → Diameter (mm)                            │
│  Using: mm_per_pixel calibration factor                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│            SIZE CATEGORIZATION & WEIGHT ESTIMATION              │
│                                                                 │
│  Size: Small (<50mm), Medium (50-60mm), Large (>60mm)        │
│  Weight: W = 0.05 × D³ (where D is diameter in mm)           │
│  Range: 30-80g (typical chicken eggs)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌────────┐   ┌─────────┐   ┌──────────┐
   │ DISPLAY │   │ LOGGING │   │ COUNTING │
   │ (Video) │   │ (CSV)   │   │ (Unique) │
   └────────┘   └─────────┘   └──────────┘
        │             │             │
        └─────────────┴─────────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │   Display Overlay Info:  │
         │ Count | Size Distribution│
         │ Weight | Class | FPS     │
         └──────────────────────────┘
```

## Data Flow: Single Egg Detection

```
Frame arrives
    │
    ├─ YOLO runs inference on entire frame
    │   ├─ Output: x1=100, y1=50, x2=200, y2=150, confidence=0.82
    │   └─ Passes threshold (0.82 > 0.65) ✓
    │
    ├─ Object Tracker processes detection
    │   ├─ Centroid: (150, 100)
    │   ├─ No existing tracked eggs, so:
    │   ├─ Assign new ID=0
    │   ├─ is_new=True (count this egg!)
    │   └─ egg_count++
    │
    ├─ CNN Classification
    │   ├─ Crop egg region: frame[50:150, 100:200]
    │   ├─ Predict: class=1 (Not Damaged), confidence=0.93
    │   ├─ Passes CNN threshold (0.93 > 0.6) ✓
    │   └─ Classification: "Not Damaged"
    │
    ├─ Size & Weight Calculation
    │   ├─ Diameter (pixels) = max(200-100, 150-50) = 100
    │   ├─ Diameter (mm) = 100 × 0.5 = 50 mm (with calibration)
    │   ├─ Size category: "Medium" (50-60mm range)
    │   └─ Weight = 0.05 × 50³ = 62.5g
    │
    └─ Output
       ├─ Display: Green box with "YOLO: 0.82", "CNN: 0.93"
       ├─ Log to CSV: [timestamp, id=0, x1=100, ..., weight=62.5]
       └─ Update count: 1 egg
```

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         main()                                   │
│                      Live View Loop                              │
└────────┬──────────────────────┬──────────────────────┬───────────┘
         │                      │                      │
         ▼                      ▼                      ▼
  ┌────────────────┐    ┌─────────────────┐   ┌──────────────────┐
  │ YOLO Model     │    │ ObjectTracker   │   │ CameraCalibrator │
  │ (YOLOv8)       │    │                 │   │                  │
  │                │    │ ─ update()      │   │ ─ pixels_to_mm() │
  │ ─ predict()    │    │ ─ track objects │   │ ─ categorize()   │
  │ ─ return boxes │    │ ─ assign IDs    │   │ ─ estimate_wt()  │
  └────────────────┘    └─────────────────┘   └──────────────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ CNN Classifier         │
                    │ (EggGradingCNN)        │
                    │                        │
                    │ ─ predict_image()      │
                    │ ─ return class + conf  │
                    └────────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ StatisticsLogger       │
                    │                        │
                    │ ─ log_detection()      │
                    │ ─ write to CSV         │
                    └────────────────────────┘
```

## Configuration Flow

```
config/config.yaml
    │
    ├─ [calibration]
    │  ├─ mm_per_pixel: 0.5  ──┬─→ CameraCalibrator
    │  └─ camera_distance: 15  │
    │                           │
    ├─ [detection]              │
    │  ├─ yolo_confidence: 0.65 ──┬─→ main() YOLO threshold
    │  └─ cnn_confidence: 0.6   ──┼─→ main() CNN threshold
    │                           │
    └─ [tracking]              │
       ├─ max_distance: 50     ──┴─→ ObjectTracker.__init__()
       └─ max_disappeared: 30  ───→ ObjectTracker.__init__()

Runtime adjustments:
    Keyboard input
       │
       ├─ '+' key ──→ yolo_conf += 0.05
       ├─ '-' key ──→ yolo_conf -= 0.05
       └─ 'C' key ──→ enable_cnn_filtering = not enable_cnn_filtering
```

## CSV Output Schema

```
egg_statistics.csv structure:

timestamp (ISO 8601)
    └─ 2026-04-07T14:35:22.123456

egg_id (from ObjectTracker)
    └─ 0, 1, 2, ... (unique per egg)

Bounding Box (pixels)
    ├─ x1: 100 (left edge)
    ├─ y1: 50  (top edge)
    ├─ x2: 200 (right edge)
    └─ y2: 150 (bottom edge)

YOLO Detection
    └─ yolo_confidence: 0.8532

CNN Classification
    ├─ class: "Damaged" or "Not Damaged"
    └─ cnn_confidence: 0.9123

Size Measurements
    ├─ diameter_px: 100 (pixels)
    ├─ diameter_mm: 50.0 (calibrated)
    ├─ size_category: "Medium"
    └─ weight_g: 62.5

Can be analyzed:
    SELECT size_category, AVG(weight_g), COUNT(*)
    FROM egg_statistics
    GROUP BY size_category
```

## Keyboard Control Flow

```
User Input
    │
    ├─ 'Q' ──→ break loop → cap.release() → exit
    ├─ 'P' ──→ paused = not paused
    ├─ 'S' ──→ cv2.imwrite(screenshot)
    ├─ '+' ──→ yolo_conf = min(0.99, yolo_conf + 0.05)
    ├─ '-' ──→ yolo_conf = max(0.1, yolo_conf - 0.05)
    └─ 'C' ──→ enable_cnn_filtering = not enable_cnn_filtering
                    │
                    └─ affects detection filtering immediately
```

## Memory & State Management

```
Persistent State (across frames):
    │
    ├─ ObjectTracker
    │  ├─ objects: {0: (cx, cy, frame_count), 1: (cx, cy, ...), ...}
    │  ├─ disappeared: {0: 0, 1: 2, ...}
    │  └─ next_id: 5 (counter for new eggs)
    │
    ├─ CameraCalibrator
    │  └─ mm_per_pixel: 0.5 (loaded once from config)
    │
    ├─ StatisticsLogger
    │  └─ log_file (opened file handle to CSV)
    │
    └─ Counters
       ├─ egg_count: 12 (total eggs in session)
       ├─ counted_ids: {0, 1, 2, 3, 4} (prevents double counting)
       └─ fps: 28.5 (exponential moving average)

Per-frame Temporary (cleared each frame):
    ├─ egg_boxes: [(x1, y1, x2, y2), ...]
    ├─ egg_confs: [0.82, 0.91, ...]
    ├─ egg_classifications: ["Not Damaged", "Damaged", ...]
    └─ size_summary: "2 Medium, 1 Large"
```

## Performance Characteristics

```
Per Frame (30 FPS = 33ms per frame):
    │
    ├─ YOLO inference: ~20-30ms (GPU) or 30-50ms (CPU)
    ├─ Tracking: ~1ms (linear in object count)
    ├─ CNN inference: ~5-10ms × N eggs
    ├─ Calibration: <1ms
    ├─ CSV logging: <1ms
    ├─ Rendering: ~5-10ms
    │
    └─ Total: ~28-35ms (typically maintains 28-30 FPS)

Bottleneck: YOLO inference (takes ~80-90% of time)

Memory usage:
    ├─ YOLO model: ~200MB
    ├─ CNN model: ~5MB
    ├─ Tracker state: ~10-100 bytes per object
    ├─ CSV buffer: ~1KB (auto-flushed per detection)
    │
    └─ Total: ~220MB typical
```

## Error Handling

```
Video Source Failure
    └─ cap.isOpened() check → print error → return

Model Loading Failure
    └─ try-except around YOLO load → print error → return

Empty Crops
    └─ egg_crop.size > 0 check → skip CNN classification

Invalid Predictions
    └─ Exception handling → mark as "Error"

CSV Write Failure
    └─ try-except in logger → print warning, continue

Invalid Config
    └─ yaml.safe_load() → use default values
```

## Integration Points for External Systems

```
Input Integration:
    ├─ Video source: OpenCV VideoCapture (camera or file)
    ├─ Configuration: YAML file (human-readable)
    └─ Calibration: Interactive tool or manual config entry

Output Integration:
    ├─ Real-time display: OpenCV imshow() window
    ├─ CSV export: Standard CSV format (readable by Excel, pandas)
    ├─ Screenshots: JPEG files with timestamp
    ├─ Console: Standard output (can be redirected)
    └─ Statistics: In-memory state (accessible via Python import)

Extension Points:
    ├─ Custom ObjectTracker: replace update() method
    ├─ Custom Calibrator: extend CameraCalibrator class
    ├─ Custom Logger: implement StatisticsLogger interface
    └─ Custom Display: modify overlay_text in main()
```

## Testing Strategy

```
Unit Tests (should implement):
    ├─ ObjectTracker.update() with various scenarios
    ├─ CameraCalibrator conversion formulas
    ├─ StatisticsLogger CSV format correctness
    └─ predict_image() confidence score ranges

Integration Tests (should implement):
    ├─ Full pipeline with mock YOLO detections
    ├─ File I/O (config loading, CSV writing)
    ├─ Keyboard control responsiveness
    └─ FPS stability under load

Manual Tests (already done):
    ├─ Syntax validation ✓
    ├─ Import validation ✓
    ├─ Configuration loading ✓
    ├─ Backward compatibility ✓
    │
    └─ Still needed:
       ├─ Live camera detection
       ├─ Object tracking accuracy
       ├─ Weight calculation accuracy
       └─ CSV correctness
```

## Deployment Checklist

```
Before production deployment:
    □ Run all unit tests
    □ Test with real camera feed
    □ Calibrate camera for specific distance
    □ Verify CSV logging works
    □ Test keyboard controls
    □ Benchmark performance on target hardware
    □ Document system configuration
    □ Set up error monitoring/logging
    □ Create backup of trained models
    □ Document troubleshooting procedures
```
