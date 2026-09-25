"""Validate and freeze a YOLO detection dataset.

The validator checks image/label pairing, YOLO box validity, exact duplicate
leakage, and perceptual near-duplicate candidates across splits. It also writes
a manifest and JSON report containing the dataset and annotation metadata.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml
from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def resolve_dataset_root(config_path: Path, config: dict) -> Path:
    configured = Path(str(config.get("path", ".")))
    if configured.is_absolute():
        return configured.resolve()
    from_working_directory = (Path.cwd() / configured).resolve()
    if (from_working_directory / "images").is_dir():
        return from_working_directory
    return (config_path.parent / configured).resolve()


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as source:
        resized = source.convert("L").resize((9, 8))
        pixels = list(resized.getdata())
    differences = [
        pixels[row * 9 + column + 1] > pixels[row * 9 + column]
        for row in range(8)
        for column in range(8)
    ]
    value = 0
    for bit in differences:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def iter_images(directory: Path) -> Iterable[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ) if directory.is_dir() else []


def validate_box(
    line: str, path: Path, line_number: int, class_count: int
) -> Tuple[int, float, float, float, float]:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
    try:
        class_id = int(fields[0])
        values = tuple(float(value) for value in fields[1:])
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: non-numeric YOLO value") from exc
    if not 0 <= class_id < class_count:
        raise ValueError(f"{path}:{line_number}: class {class_id} is outside configured classes")
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError(f"{path}:{line_number}: coordinates must be in [0, 1]")
    _, _, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}:{line_number}: box width and height must be positive")
    return class_id, *values


def write_preview(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    class_names: List[str],
) -> int:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        box_count = 0
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            class_id, x_center, y_center, box_width, box_height = map(
                float, line.split()
            )
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)
            color = "red" if int(class_id) == 1 else "green"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, width // 400))
            draw.text((max(0, x1 + 5), max(0, y1 + 5)), class_names[int(class_id)], fill=color)
            box_count += 1
        image.thumbnail((1200, 1200))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return box_count


def validate(args: argparse.Namespace) -> dict:
    config_path = Path(args.data_yaml).resolve()
    config = yaml.safe_load(config_path.read_text())
    names = config.get("names", {})
    class_names = [
        str(names[index] if isinstance(names, list) else names.get(index, names.get(str(index), index)))
        for index in range(int(config["nc"]))
    ]
    dataset_root = resolve_dataset_root(config_path, config)
    records: List[dict] = []
    errors: List[str] = []
    split_hashes: Dict[str, Dict[str, Path]] = {}
    perceptual_hashes: List[Tuple[str, Path, int]] = []
    class_counts = {split: Counter() for split in SPLITS}
    oversized_boxes = []

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        split_hashes[split] = {}
        images = list(iter_images(image_dir))
        if not images:
            errors.append(f"{split}: no images found in {image_dir}")
        image_names = {path.stem for path in images}
        label_names = {
            path.stem for path in label_dir.glob("*.txt")
        } if label_dir.is_dir() else set()
        for missing in sorted(image_names - label_names):
            errors.append(f"{split}: missing label for {missing}")
        for orphan in sorted(label_names - image_names):
            errors.append(f"{split}: label has no image for {orphan}")

        for image_path in images:
            relative_image = image_path.relative_to(dataset_root).as_posix()
            label_path = label_dir / f"{image_path.stem}.txt"
            try:
                with Image.open(image_path) as decoded:
                    decoded.verify()
                    width, height = decoded.size
                digest = image_hash(image_path)
                perceptual = perceptual_hash(image_path)
                split_hashes[split][digest] = image_path
                perceptual_hashes.append((split, image_path, perceptual))
                boxes = []
                if label_path.is_file():
                    for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
                        if line.strip():
                            box = validate_box(line, label_path, line_number, len(class_names))
                            boxes.append(box)
                            class_counts[split][class_names[box[0]]] += 1
                            if box[3] * box[4] >= args.oversized_area_threshold:
                                oversized_boxes.append({
                                    "split": split,
                                    "image": relative_image,
                                    "class": class_names[box[0]],
                                    "area_fraction": round(box[3] * box[4], 6),
                                })
                if not boxes and args.require_boxes:
                    errors.append(f"{split}: no valid boxes for {relative_image}")
                records.append({
                    "split": split,
                    "image": relative_image,
                    "label": label_path.relative_to(dataset_root).as_posix(),
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "box_count": len(boxes),
                    "classes": ",".join(class_names[box[0]] for box in boxes),
                    "annotation_source": args.annotation_source,
                })
            except (OSError, ValueError) as exc:
                errors.append(str(exc))

    duplicate_groups = []
    all_hashes = {}
    for split, hashes in split_hashes.items():
        for digest, path in hashes.items():
            all_hashes.setdefault(digest, []).append((split, path.relative_to(dataset_root).as_posix()))
    for digest, paths in all_hashes.items():
        if len({split for split, _ in paths}) > 1:
            duplicate_groups.append({"sha256": digest, "files": paths})

    near_duplicates = []
    for index, (left_split, left_path, left_hash) in enumerate(perceptual_hashes):
        for right_split, right_path, right_hash in perceptual_hashes[index + 1:]:
            if left_split != right_split and hamming_distance(left_hash, right_hash) <= args.near_duplicate_threshold:
                near_duplicates.append({
                    "distance": hamming_distance(left_hash, right_hash),
                    "left": left_path.relative_to(dataset_root).as_posix(),
                    "right": right_path.relative_to(dataset_root).as_posix(),
                })

    report = {
        "dataset_root": str(dataset_root),
        "data_yaml": str(config_path),
        "classes": class_names,
        "splits": {
            split: {
                "images": sum(record["split"] == split for record in records),
                "boxes": sum(record["split"] == split for record in records for _ in range(record["box_count"])),
                "class_counts": dict(class_counts[split]),
            }
            for split in SPLITS
        },
        "annotation_source": args.annotation_source,
        "duplicate_groups": duplicate_groups,
        "near_duplicate_candidates": near_duplicates,
        "oversized_boxes": oversized_boxes,
        "errors": errors,
        "valid": not errors and not duplicate_groups and not near_duplicates,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=records[0].keys() if records else [
                "split", "image", "label", "sha256", "width", "height",
                "box_count", "classes", "annotation_source",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    if args.preview_dir:
        preview_root = Path(args.preview_dir)
        for split in SPLITS:
            candidates = [
                Path(dataset_root / record["image"])
                for record in records
                if record["split"] == split
            ]
            random.Random(args.preview_seed).shuffle(candidates)
            for image_path in candidates[:args.preview_per_split]:
                write_preview(
                    image_path,
                    dataset_root / "labels" / split / f"{image_path.stem}.txt",
                    preview_root / split / image_path.name,
                    class_names,
                )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", default="data/detection/data.yaml")
    parser.add_argument("--manifest", default="data/detection/manifest.csv")
    parser.add_argument("--report", default="reports/detection_dataset_report.json")
    parser.add_argument(
        "--annotation-source",
        default="heuristic-contour-generated; requires visual verification",
    )
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    parser.add_argument(
        "--oversized-area-threshold",
        type=float,
        default=0.75,
        help="Report boxes covering at least this fraction of the image",
    )
    parser.add_argument(
        "--preview-dir",
        help="Write sampled images with rendered YOLO boxes to this directory",
    )
    parser.add_argument("--preview-per-split", type=int, default=20)
    parser.add_argument("--preview-seed", type=int, default=42)
    parser.add_argument(
        "--require-boxes",
        action="store_true",
        help="Reject empty-tray annotations; empty label files are valid by default",
    )
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
