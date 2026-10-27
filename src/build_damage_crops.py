"""Build detector-cropped damage evaluation images from a manifest.

This creates an evaluation set that exercises the real detector -> crop ->
damage-classifier path. It is not a replacement for the full-image test set.
Missed detections are recorded and omitted from the crop manifest.
"""

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:100]


def padded_crop(image, bbox, padding: float):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    pad_x = int(round((x2 - x1) * padding))
    pad_y = int(round((y2 - y1) * padding))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right], (left, top, right, bottom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/damage/manifests_v2/test.csv")
    )
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("egg_detection/finetune_egg_v1/weights/best.pt"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/damage_crops/test")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/damage_crop_test_report.json"),
    )
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--min-crop-size", type=int, default=8)
    parser.add_argument("--max-detections-per-image", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Missing detector: {args.model}")
    if args.padding < 0:
        raise ValueError("padding must be non-negative")
    if args.max_detections_per_image < 1:
        raise ValueError("max-detections-per-image must be at least 1")

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {args.manifest}")
    if "filename" not in rows[0] or "label" not in rows[0]:
        raise ValueError("Manifest must contain filename and label columns")
    if args.limit is not None:
        rows = rows[: args.limit]

    manifest_rows = []
    missed = []
    multi_detection = []
    failed_read = []
    label_counts = Counter()
    source_counts = Counter()
    detection_confidences = []
    total_detections = 0
    detected_images = 0
    usable_images = 0

    for index, row in enumerate(rows, start=1):
        source_path = args.image_root / row["filename"]
        image = cv2.imread(str(source_path))
        if image is None:
            failed_read.append(str(source_path))
            continue
        predict_kwargs = {
            "source": image,
            "conf": args.confidence,
            "verbose": False,
        }
        if args.device:
            predict_kwargs["device"] = args.device
        result = model.predict(**predict_kwargs)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            missed.append(str(source_path))
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        detected_images += 1
        if len(xyxy) > args.max_detections_per_image:
            multi_detection.append(
                {"source_image": str(source_path), "detections": len(xyxy)}
            )
            continue
        usable_images += 1
        source_id = row.get("source_id", "unknown")
        for egg_index, (bbox, confidence) in enumerate(zip(xyxy, confs), start=1):
            crop_result = padded_crop(image, bbox, args.padding)
            if crop_result is None:
                continue
            crop, crop_bbox = crop_result
            crop_height, crop_width = crop.shape[:2]
            if crop_width < args.min_crop_size or crop_height < args.min_crop_size:
                continue
            crop_name = (
                f"{safe_stem(source_path)}__egg{egg_index:02d}.jpg"
            )
            crop_path = args.output / crop_name
            if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Failed to write crop: {crop_path}")
            manifest_rows.append(
                {
                    "filename": str(crop_path),
                    "label": row["label"],
                    "source_image": str(source_path),
                    "source_id": source_id,
                    "egg_index": egg_index,
                    "detection_confidence": f"{float(confidence):.6f}",
                    "bbox_x1": crop_bbox[0],
                    "bbox_y1": crop_bbox[1],
                    "bbox_x2": crop_bbox[2],
                    "bbox_y2": crop_bbox[3],
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                }
            )
            label_counts[row["label"]] += 1
            source_counts[source_id] += 1
            detection_confidences.append(float(confidence))
            total_detections += 1

        if index % 25 == 0:
            print(f"processed {index}/{len(rows)} images")

    manifest_path = args.output / "manifest.csv"
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else [
        "filename",
        "label",
        "source_image",
        "source_id",
        "egg_index",
        "detection_confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_width",
        "crop_height",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    confidence_summary = {}
    if detection_confidences:
        confidence_summary = {
            "min": round(min(detection_confidences), 6),
            "max": round(max(detection_confidences), 6),
            "mean": round(
                sum(detection_confidences) / len(detection_confidences), 6
            ),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "detector": str(args.model),
        "output": str(args.output),
        "confidence_threshold": args.confidence,
        "padding": args.padding,
        "min_crop_size": args.min_crop_size,
        "source_images": len(rows),
        "detected_images": detected_images,
        "usable_images": usable_images,
        "missed_images": len(missed),
        "failed_read_images": failed_read,
        "detection_coverage": (
            round(detected_images / len(rows), 6) if rows else 0.0
        ),
        "usable_coverage": round(usable_images / len(rows), 6) if rows else 0.0,
        "multi_detection_skipped": len(multi_detection),
        "crops": total_detections,
        "label_counts": dict(sorted(label_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "multi_detection_images": multi_detection,
        "detection_confidence": confidence_summary,
        "crop_manifest": str(manifest_path),
        "limitations": [
            "Crops inherit the source image label.",
            "Missed detections are excluded from the crop manifest.",
            "This set is for pipeline evaluation, not detector mAP.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"source images: {len(rows)}")
    print(f"detected images: {detected_images}")
    print(f"usable images: {usable_images}")
    print(f"missed images: {len(missed)}")
    print(f"multi-detection skipped: {len(multi_detection)}")
    print(f"crops: {total_detections}")
    print(f"label counts: {dict(sorted(label_counts.items()))}")
    print(f"manifest: {manifest_path}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
