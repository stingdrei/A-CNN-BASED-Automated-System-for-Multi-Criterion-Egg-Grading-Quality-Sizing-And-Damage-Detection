"""Evaluate a YOLO egg detector on the locked test split.

The Ultralytics validation metrics are supplemented with tray-level count
metrics. Threshold selection must happen on validation data before this script
is run against the test split.
"""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

try:
    from src.tray_count import deduplicate_boxes, tray_count_status
except ModuleNotFoundError:  # Support `python src/evaluate_detector.py`.
    from tray_count import deduplicate_boxes, tray_count_status


def read_box_count(label_path: Path) -> int:
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    return sum(bool(line.strip()) for line in label_path.read_text().splitlines())


def evaluate(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    data_path = Path(args.data_yaml).resolve()
    config = yaml.safe_load(data_path.read_text())
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()
    runtime_config = dict(config)
    runtime_config["path"] = str(dataset_root)
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    model = YOLO(str(Path(args.model).resolve()))
    with NamedTemporaryFile("w", suffix=".yaml", delete=False) as runtime_file:
        yaml.safe_dump(runtime_config, runtime_file)
        runtime_data_path = runtime_file.name
    try:
        validation = model.val(
            data=runtime_data_path,
            split=args.split,
            conf=args.confidence,
            plots=False,
            verbose=False,
        )
    finally:
        Path(runtime_data_path).unlink(missing_ok=True)
    exact_counts = 0
    false_positives = 0
    missed = 0
    total_truth = 0
    duplicate_boxes_removed = 0
    empty_trays = 0
    over_capacity_trays = 0
    truth_empty_trays = 0
    truth_over_capacity_trays = 0
    zero_to_five_images = 0
    zero_to_five_exact = 0
    zero_to_five_false_positives = 0
    zero_to_five_missed = 0
    for image_path in image_paths:
        result = model.predict(
            source=str(image_path),
            conf=args.confidence,
            verbose=False,
            max_det=50,
        )[0]
        candidates = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for index in range(len(xyxy)):
                x1, y1, x2, y2 = [int(value) for value in xyxy[index]]
                if x2 <= x1 or y2 <= y1:
                    continue
                candidates.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(confs[index]),
                    }
                )
        candidates, removed = deduplicate_boxes(
            candidates, args.duplicate_iou
        )
        predicted = len(candidates)
        duplicate_boxes_removed += removed
        truth = read_box_count(label_dir / f"{image_path.stem}.txt")
        exact_counts += predicted == truth
        false_positives += max(0, predicted - truth)
        missed += max(0, truth - predicted)
        total_truth += truth
        count = tray_count_status(predicted, args.max_eggs)
        if predicted == 0:
            empty_trays += 1
        elif count["over_capacity"]:
            over_capacity_trays += 1
        if truth == 0:
            truth_empty_trays += 1
        elif truth > args.max_eggs:
            truth_over_capacity_trays += 1
        else:
            zero_to_five_images += 1
            zero_to_five_exact += predicted == truth
            zero_to_five_false_positives += max(0, predicted - truth)
            zero_to_five_missed += max(0, truth - predicted)

    metrics = validation.results_dict
    report = {
        "model": str(Path(args.model).resolve()),
        "data_yaml": str(data_path),
        "split": args.split,
        "confidence": args.confidence,
        "images": len(image_paths),
        "exact_tray_count_accuracy": exact_counts / len(image_paths),
        "false_positives_per_tray": false_positives / len(image_paths),
        "missed_eggs_per_tray": missed / len(image_paths),
        "duplicate_boxes_removed": duplicate_boxes_removed,
        "predicted_empty_trays": empty_trays,
        "predicted_over_capacity_trays": over_capacity_trays,
        "truth_empty_trays": truth_empty_trays,
        "truth_over_capacity_trays": truth_over_capacity_trays,
        "zero_to_five": {
            "images": zero_to_five_images,
            "exact_tray_count_accuracy": (
                zero_to_five_exact / zero_to_five_images
                if zero_to_five_images
                else None
            ),
            "false_positives_per_tray": (
                zero_to_five_false_positives / zero_to_five_images
                if zero_to_five_images
                else None
            ),
            "missed_eggs_per_tray": (
                zero_to_five_missed / zero_to_five_images
                if zero_to_five_images
                else None
            ),
        },
        "max_tray_eggs": args.max_eggs,
        "duplicate_iou": args.duplicate_iou,
        "ground_truth_eggs": total_truth,
        "ultralytics_metrics": {key: float(value) for key, value in metrics.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-yaml", default="data/detection/data.yaml")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--output", default="reports/detector_test_report.json")
    parser.add_argument("--max-eggs", type=int, default=5)
    parser.add_argument("--duplicate-iou", type=float, default=0.85)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
