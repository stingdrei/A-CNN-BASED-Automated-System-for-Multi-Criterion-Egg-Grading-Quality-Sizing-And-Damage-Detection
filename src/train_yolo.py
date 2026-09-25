"""
Egg Detection YOLO Training Script
Usage:
1. Prepare dataset (or use convert_to_yolo.py)
2. Run: python train_yolo.py --train
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "detection"
YOLO_MODEL = "yolov8s.pt"
IMG_SIZE = 640
EPOCHS = 100
BATCH = 16
PROJECT_NAME = "egg_detection"
RUN_NAME = "train1"


def get_data_config() -> dict:
    yaml_path = DATASET_DIR / "data.yaml"
    if yaml_path.is_file():
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
            cfg["path"] = str(DATASET_DIR)
            return cfg
    return {
        "path": str(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": ["B-eggs", "W-eggs"],
    }


def create_data_yaml():
    data_config = get_data_config()
    with open(DATASET_DIR / "data.yaml", "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print("Created data/detection/data.yaml")


def data_config() -> dict:
    return get_data_config()


def prepare_directories():
    dirs = [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "images" / "test",
        DATASET_DIR / "labels" / "train",
        DATASET_DIR / "labels" / "val",
        DATASET_DIR / "labels" / "test",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Created dataset directories.")


def validate_dataset():
    """Fail early with an actionable message when images are unavailable."""
    missing_splits = []
    for split in ("train", "val", "test"):
        image_dir = DATASET_DIR / "images" / split
        image_count = sum(
            1
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if image_dir.is_dir() else 0
        if image_count == 0:
            missing_splits.append(f"{split}: {image_dir}")

    if missing_splits:
        details = "\n".join(f"  - {item}" for item in missing_splits)
        raise FileNotFoundError(
            "YOLO training cannot start because no detector images were found:\n"
            f"{details}\n"
            "Run `python src/convert_to_yolo.py` after placing source images in "
            "`data/damage/`, or restore the tracked `data/detection/images/` files."
        )


def clean_cache_files():
    labels_dir = DATASET_DIR / "labels"
    if labels_dir.is_dir():
        for cache_file in labels_dir.glob("*.cache"):
            try:
                cache_file.unlink()
                print(f"Removed cache file: {cache_file.name}")
            except Exception as e:
                print(f"Failed to remove cache file {cache_file}: {e}")


def train_yolo(device="cpu"):
    clean_cache_files()
    create_data_yaml()
    validate_dataset()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        return

    with NamedTemporaryFile("w", suffix=".yaml", delete=False) as runtime_file:
        yaml.safe_dump(data_config(), runtime_file)
        runtime_data_path = runtime_file.name
    try:
        model = YOLO(YOLO_MODEL)
        results = model.train(
            data=runtime_data_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            workers=0,
            project=os.path.abspath(PROJECT_NAME),
            name=RUN_NAME,
            device=device,
            exist_ok=True,
            pretrained=True,
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            box=7.5,
            cls=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            patience=50,
            verbose=True,
            seed=0,
            deterministic=True,
        )
    finally:
        Path(runtime_data_path).unlink(missing_ok=True)
    print("========================")
    print("\nTraining complete!")
    print(f"Results: {results}")
    print(f"Best model: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print("========================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare", action="store_true", help="Prepare directories only"
    )
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument(
        "--device", default="cpu", help="Device: cpu, cuda:0, mps (default: cpu)"
    )
    args = parser.parse_args()

    if args.prepare:
        prepare_directories()
        create_data_yaml()
    elif args.train:
        train_yolo(device=args.device)
    else:
        prepare_directories()
        create_data_yaml()
        print("\nTo train: python train_yolo.py --train")
