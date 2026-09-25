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


def get_data_config(dataset_dir: Path = DATASET_DIR) -> dict:
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.is_file():
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
            cfg["path"] = str(dataset_dir.resolve())
            return cfg
    return {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["egg"],
    }


def create_data_yaml(dataset_dir: Path = DATASET_DIR):
    data_config = get_data_config(dataset_dir)
    with open(dataset_dir / "data.yaml", "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print(f"Created {dataset_dir / 'data.yaml'}")


def data_config(dataset_dir: Path = DATASET_DIR) -> dict:
    return get_data_config(dataset_dir)


def prepare_directories(dataset_dir: Path = DATASET_DIR):
    dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "images" / "test",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
        dataset_dir / "labels" / "test",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Created dataset directories.")


def validate_dataset(dataset_dir: Path = DATASET_DIR):
    """Fail early with an actionable message when images are unavailable."""
    missing_splits = []
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
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
            "Prepare the dataset before training, or restore the required image files."
        )


def clean_cache_files(dataset_dir: Path = DATASET_DIR):
    labels_dir = dataset_dir / "labels"
    if labels_dir.is_dir():
        for cache_file in labels_dir.glob("*.cache"):
            try:
                cache_file.unlink()
                print(f"Removed cache file: {cache_file.name}")
            except Exception as e:
                print(f"Failed to remove cache file {cache_file}: {e}")


def train_yolo(
    device="cpu",
    model_path=YOLO_MODEL,
    dataset_dir=DATASET_DIR,
    project=PROJECT_NAME,
    run_name=RUN_NAME,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
):
    checkpoint = Path(model_path)
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Starting checkpoint not found: {checkpoint}")
    clean_cache_files(dataset_dir)
    create_data_yaml(dataset_dir)
    validate_dataset(dataset_dir)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        return

    with NamedTemporaryFile("w", suffix=".yaml", delete=False) as runtime_file:
        yaml.safe_dump(data_config(dataset_dir), runtime_file)
        runtime_data_path = runtime_file.name
    try:
        model = YOLO(str(checkpoint))
        results = model.train(
            data=runtime_data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=0,
            project=str((REPO_ROOT / project).resolve())
            if not os.path.isabs(project)
            else project,
            name=run_name,
            device=device,
            exist_ok=False,
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
    print(f"Best model: {project}/{run_name}/weights/best.pt")
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
    parser.add_argument("--model", default=YOLO_MODEL, help="Starting YOLO checkpoint")
    parser.add_argument("--data-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--project", default=PROJECT_NAME)
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()

    if args.prepare:
        prepare_directories(args.data_root)
        create_data_yaml(args.data_root)
    elif args.train:
        train_yolo(
            device=args.device,
            model_path=args.model,
            dataset_dir=args.data_root,
            project=args.project,
            run_name=args.name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
        )
    else:
        prepare_directories(args.data_root)
        create_data_yaml(args.data_root)
        print("\nTo train: python train_yolo.py --train")
