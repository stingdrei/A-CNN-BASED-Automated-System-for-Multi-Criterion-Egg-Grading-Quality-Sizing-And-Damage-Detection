"""Audit COCO datasets for a binary Clean/Stained cleanliness classifier.

This script is read-only with respect to the source datasets. It:

* counts images, annotations, classes, and viable Clean/Stained crops;
* validates COCO boxes and reports small/invalid crops;
* checks exact and perceptual duplicate leakage;
* flags stock/watermark-prone filenames for manual review;
* writes annotated contact sheets and crop previews;
* writes a candidate crop manifest for later dataset generation.

It does not write crop files or train a model.

Example:
    python src/audit_cleanliness_sources.py \
      --source "clean eggs" \
      --source "Eggs degrading quality" \
      --report reports/cleanliness_source_audit.json \
      --manifest reports/cleanliness_candidate_crops.csv \
      --preview-dir reports/cleanliness_source_previews
"""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
LABEL_NAMES = {0: "Clean", 1: "Stained"}
STOCK_KEYWORDS = (
    "dreamstime",
    "depositphotos",
    "shutterstock",
    "istock",
    "gettyimages",
    "alamy",
    "123rf",
    "stock-photo",
    "stockphoto",
    "watermark",
    "adobe",
    "adobestock",
    "1000_f_",
    "240_f_",
    "360_f_",
    "600nw",
    "260nw",
)

# Category names are normalized to lower-case, single-space strings.
CATEGORY_RULES = {
    "brownegg": (0, "Clean", False),
    "lightbrownegg": (0, "Clean", False),
    "whiteegg": (0, "Clean", False),
    "white egg": (0, "Clean", True),
    "dirt stained brown egg": (1, "Stained", False),
    "dirt stained white egg": (1, "Stained", False),
}
UNMAPPED_CATEGORIES = {"color", "eggs"}


def normalize_category(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def label_for_category(name: str) -> Optional[Tuple[int, str, bool]]:
    """Return (label, label_name, weak_label) or None for unmapped classes."""
    normalized = normalize_category(name)
    if normalized in UNMAPPED_CATEGORIES:
        return None
    return CATEGORY_RULES.get(normalized)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "dataset"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: Image.Image) -> int:
    resized = image.convert("L").resize((9, 8))
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


def box_to_xyxy(
    bbox: Sequence[float], width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    """Convert a COCO xywh box to clipped integer xyxy, or None if invalid."""
    if len(bbox) != 4:
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if box_width <= 0 or box_height <= 0:
        return None
    left = max(0, min(width, int(round(x))))
    top = max(0, min(height, int(round(y))))
    right = max(0, min(width, int(round(x + box_width))))
    bottom = max(0, min(height, int(round(y + box_height))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def numeric_summary(values: Iterable[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "median": round(median, 3),
        "mean": round(sum(ordered) / len(ordered), 3),
        "max": round(ordered[-1], 3),
    }


def stock_keyword(filename: str) -> Optional[str]:
    lowered = filename.lower()
    for keyword in STOCK_KEYWORDS:
        if keyword in lowered:
            return keyword
    return None


def load_split(root: Path, split: str) -> Tuple[dict, List[str]]:
    annotation_path = root / split / "_annotations.coco.json"
    errors: List[str] = []
    if not annotation_path.is_file():
        return {"images": [], "annotations": [], "categories": {}}, [
            f"missing COCO annotations: {annotation_path}"
        ]
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"images": [], "annotations": [], "categories": {}}, [
            f"failed to read {annotation_path}: {exc}"
        ]
    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in payload.get("categories", [])
    }
    return {
        "images": payload.get("images", []),
        "annotations": payload.get("annotations", []),
        "categories": categories,
    }, errors


def audit_source(
    root: Path, min_crop_size: int
) -> Tuple[dict, List[dict], List[dict]]:
    """Audit one COCO source and return report, image records, and crop records."""
    dataset = root.name
    report = {
        "root": str(root),
        "splits": {},
        "images": 0,
        "annotations": 0,
        "category_counts": {},
        "mapped_counts": {"Clean": 0, "Stained": 0},
        "weak_label_counts": {"Clean": 0, "Stained": 0},
        "mapped_by_split": {},
        "unmapped_categories": {},
        "invalid_boxes": [],
        "missing_images": [],
        "image_errors": [],
        "stock_keyword_images": [],
        "image_dimensions": {},
        "crop_dimensions": {},
        "small_crops": 0,
        "empty_images": 0,
    }
    image_records: List[dict] = []
    crop_records: List[dict] = []
    all_category_counts: Counter = Counter()
    all_mapped_counts: Counter = Counter()
    weak_counts: Counter = Counter()
    image_widths: List[int] = []
    image_heights: List[int] = []
    crop_widths: List[int] = []
    crop_heights: List[int] = []
    split_categories: Dict[str, List[str]] = {}

    for split in SPLITS:
        payload, split_errors = load_split(root, split)
        report["image_errors"].extend(split_errors)
        images = payload["images"]
        annotations = payload["annotations"]
        categories = payload["categories"]
        split_categories[split] = sorted(categories.values())
        by_image: Dict[int, List[dict]] = defaultdict(list)
        for annotation in annotations:
            by_image[int(annotation.get("image_id", -1))].append(annotation)

        split_category_counts: Counter = Counter()
        split_mapped_counts: Counter = Counter()
        split_mapped_by_label: Counter = Counter()
        split_images_with_annotations = 0
        split_missing = 0
        split_invalid = 0

        for image_row in images:
            image_id = int(image_row.get("id", -1))
            file_name = str(image_row.get("file_name", ""))
            image_path = root / split / file_name
            if not image_path.is_file():
                report["missing_images"].append(str(image_path))
                split_missing += 1
                continue

            try:
                with Image.open(image_path) as decoded:
                    decoded.verify()
                with Image.open(image_path) as decoded:
                    image = decoded.convert("RGB")
                    width, height = image.size
                    digest = sha256_file(image_path)
                    perceptual = perceptual_hash(image)
            except (OSError, ValueError) as exc:
                report["image_errors"].append(f"{image_path}: {exc}")
                continue

            image_widths.append(width)
            image_heights.append(height)
            stock = stock_keyword(file_name)
            if stock:
                report["stock_keyword_images"].append(
                    {"image": str(image_path), "keyword": stock}
                )

            image_annotations = by_image.get(image_id, [])
            image_box_count = len(image_annotations)
            if image_box_count:
                split_images_with_annotations += 1
            else:
                report["empty_images"] += 1

            image_record = {
                "dataset": dataset,
                "split": split,
                "image_id": image_id,
                "path": str(image_path),
                "relative": f"{split}/{file_name}",
                "width": width,
                "height": height,
                "sha256": digest,
                "phash": perceptual,
                "stock_keyword": stock,
                "annotations": image_annotations,
                "categories": categories,
            }
            image_records.append(image_record)

            for annotation in image_annotations:
                category_id = int(annotation.get("category_id", -1))
                category = str(categories.get(category_id, category_id))
                split_category_counts[category] += 1
                mapping = label_for_category(category)
                bbox = annotation.get("bbox", [])
                xyxy = box_to_xyxy(bbox, width, height) if bbox else None
                if xyxy is None:
                    report["invalid_boxes"].append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "image": str(image_path),
                            "category": category,
                            "bbox": bbox,
                        }
                    )
                    split_invalid += 1
                    continue

                x1, y1, x2, y2 = xyxy
                crop_width, crop_height = x2 - x1, y2 - y1
                crop_widths.append(crop_width)
                crop_heights.append(crop_height)
                if crop_width < min_crop_size or crop_height < min_crop_size:
                    report["small_crops"] += 1

                if mapping is None:
                    report["unmapped_categories"][category] = (
                        report["unmapped_categories"].get(category, 0) + 1
                    )
                    continue

                label, label_name, weak = mapping
                split_mapped_counts[label_name] += 1
                split_mapped_by_label[label_name] += 1
                if weak:
                    weak_counts[label_name] += 1
                crop_records.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "source_image": str(image_path),
                        "annotation_id": int(annotation.get("id", -1)),
                        "category": category,
                        "label": label,
                        "label_name": label_name,
                        "weak_label": int(weak),
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "crop_width": crop_width,
                        "crop_height": crop_height,
                        "image_width": width,
                        "image_height": height,
                        "sha256": digest,
                        "stock_keyword": stock or "",
                    }
                )

        all_category_counts.update(split_category_counts)
        all_mapped_counts.update(split_mapped_counts)
        report["splits"][split] = {
            "images": len(images),
            "images_with_annotations": split_images_with_annotations,
            "annotations": len(annotations),
            "categories": sorted(categories.values()),
            "category_counts": dict(sorted(split_category_counts.items())),
            "mapped_counts": dict(sorted(split_mapped_counts.items())),
            "missing_images": split_missing,
            "invalid_boxes": split_invalid,
        }
        report["mapped_by_split"][split] = dict(sorted(split_mapped_by_label.items()))

    report["images"] = int(
        sum(split["images"] for split in report["splits"].values())
    )
    report["annotations"] = int(
        sum(split["annotations"] for split in report["splits"].values())
    )
    report["category_counts"] = dict(sorted(all_category_counts.items()))
    report["mapped_counts"] = dict(sorted(all_mapped_counts.items()))
    report["weak_label_counts"] = dict(sorted(weak_counts.items()))
    report["image_dimensions"] = {
        "width": numeric_summary(image_widths),
        "height": numeric_summary(image_heights),
    }
    report["crop_dimensions"] = {
        "width": numeric_summary(crop_widths),
        "height": numeric_summary(crop_heights),
    }
    report["source_categories"] = split_categories
    return report, image_records, crop_records


def label_color(category: str, categories: Dict[int, str]) -> str:
    mapping = label_for_category(category)
    if mapping is None:
        return "#808080"
    return "#1a9850" if mapping[0] == 0 else "#d73027"


def draw_thumbnail(record: dict, cell_size: int = 340) -> Optional[Image.Image]:
    path = Path(record["path"])
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except OSError:
        return None
    original_width, original_height = image.size
    image.thumbnail((cell_size - 16, cell_size - 42))
    canvas = Image.new("RGB", (cell_size, cell_size), "white")
    left = (cell_size - image.width) // 2
    top = 8
    canvas.paste(image, (left, top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    scale_x = image.width / original_width
    scale_y = image.height / original_height
    for annotation in record["annotations"]:
        bbox = annotation.get("bbox", [])
        xyxy = box_to_xyxy(bbox, original_width, original_height)
        if xyxy is None:
            continue
        category = str(
            record["categories"].get(
                int(annotation.get("category_id", -1)),
                annotation.get("category_id", "?"),
            )
        )
        x1, y1, x2, y2 = xyxy
        draw.rectangle(
            (
                left + int(x1 * scale_x),
                top + int(y1 * scale_y),
                left + int(x2 * scale_x),
                top + int(y2 * scale_y),
            ),
            outline=label_color(category, record["categories"]),
            width=2,
        )
        draw.text(
            (left + int(x1 * scale_x) + 2, top + int(y1 * scale_y) + 2),
            category[:28],
            fill=label_color(category, record["categories"]),
            font=font,
        )
    caption = f"{record['split']}/{Path(record['path']).name}"[:48]
    draw.text((6, cell_size - 27), caption, fill="black", font=font)
    if record["stock_keyword"]:
        draw.text((6, cell_size - 15), f"stock:{record['stock_keyword']}", fill="#b2182b", font=font)
    return canvas


def write_contact_sheets(
    records: List[dict], preview_dir: Path, columns: int = 3, rows: int = 4
) -> List[str]:
    sheets: List[str] = []
    if not records:
        return sheets
    preview_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = columns * rows
    for index in range(0, len(records), per_sheet):
        chunk = records[index : index + per_sheet]
        canvas = Image.new(
            "RGB", (columns * 340, rows * 340), (238, 238, 238)
        )
        for offset, record in enumerate(chunk):
            thumbnail = draw_thumbnail(record)
            if thumbnail is None:
                continue
            column = offset % columns
            row = offset // columns
            canvas.paste(thumbnail, (column * 340, row * 340))
        name = f"{safe_slug(records[0]['dataset'])}_{records[0]['split']}_{index // per_sheet:02d}.jpg"
        output_path = preview_dir / name
        canvas.save(output_path, quality=90)
        sheets.append(str(output_path))
    return sheets


def write_crop_previews(
    crop_records: List[dict],
    preview_dir: Path,
    per_class: int,
    seed: int = 42,
) -> Dict[str, List[str]]:
    outputs: Dict[str, List[str]] = {}
    for label_name in LABEL_NAMES.values():
        candidates = [row for row in crop_records if row["label_name"] == label_name]
        if not candidates:
            outputs[label_name] = []
            continue
        random.Random(seed).shuffle(candidates)
        selected = candidates[:per_class]
        cells: List[Image.Image] = []
        for row in selected:
            try:
                with Image.open(row["source_image"]) as source:
                    crop = source.convert("RGB").crop(
                        (
                            row["bbox_x1"],
                            row["bbox_y1"],
                            row["bbox_x2"],
                            row["bbox_y2"],
                        )
                    )
            except OSError:
                continue
            crop = ImageOps.contain(crop, (150, 150))
            cell = Image.new("RGB", (170, 190), "white")
            cell.paste(
                crop,
                ((170 - crop.width) // 2, 5),
            )
            draw = ImageDraw.Draw(cell)
            font = ImageFont.load_default()
            draw.text((4, 160), row["category"][:26], fill="black", font=font)
            draw.text((4, 174), row["split"], fill="#555555", font=font)
            cells.append(cell)

        per_sheet = 8 * 6
        sheet_dir = preview_dir / "crops"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        sheet_paths: List[str] = []
        for index in range(0, len(cells), per_sheet):
            chunk = cells[index : index + per_sheet]
            canvas = Image.new("RGB", (8 * 170, 6 * 190), (238, 238, 238))
            for offset, cell in enumerate(chunk):
                canvas.paste(
                    cell,
                    ((offset % 8) * 170, (offset // 8) * 190),
                )
            name = f"{label_name.lower()}_{index // per_sheet:02d}.jpg"
            output_path = sheet_dir / name
            canvas.save(output_path, quality=90)
            sheet_paths.append(str(output_path))
        outputs[label_name] = sheet_paths
    return outputs


def write_manifest(crops: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "split",
        "source_image",
        "annotation_id",
        "category",
        "label",
        "label_name",
        "weak_label",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_width",
        "crop_height",
        "image_width",
        "image_height",
        "sha256",
        "stock_keyword",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(crops)


def duplicate_audit(
    records: List[dict], near_duplicate_threshold: int
) -> Tuple[List[dict], List[dict], int]:
    exact: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        exact[record["sha256"]].append(record)
    exact_groups = []
    for digest, group in exact.items():
        if len(group) < 2:
            continue
        exact_groups.append(
            {
                "sha256": digest,
                "files": [
                    {
                        "dataset": item["dataset"],
                        "split": item["split"],
                        "image": item["path"],
                    }
                    for item in group
                ],
            }
        )

    near_pairs = []
    total_near = 0
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if left["sha256"] == right["sha256"]:
                continue
            distance = hamming_distance(left["phash"], right["phash"])
            if distance > near_duplicate_threshold:
                continue
            if left["split"] == right["split"] and left["dataset"] == right["dataset"]:
                continue
            total_near += 1
            if len(near_pairs) < 200:
                near_pairs.append(
                    {
                        "distance": distance,
                        "left": f"{left['dataset']}:{left['split']}/{Path(left['path']).name}",
                        "right": f"{right['dataset']}:{right['split']}/{Path(right['path']).name}",
                    }
                )
    return exact_groups, near_pairs, total_near


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        required=True,
        help="COCO dataset root; repeat for multiple datasets",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/cleanliness_source_audit.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/cleanliness_candidate_crops.csv"),
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("reports/cleanliness_source_previews"),
    )
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    parser.add_argument("--min-crop-size", type=int, default=24)
    parser.add_argument("--crop-preview-per-class", type=int, default=80)
    args = parser.parse_args()

    if args.min_crop_size < 1:
        raise ValueError("--min-crop-size must be positive")
    if args.crop_preview_per_class < 1:
        raise ValueError("--crop-preview-per-class must be positive")

    all_image_records: List[dict] = []
    all_crop_records: List[dict] = []
    sources = {}
    preview_sheets: Dict[str, List[str]] = {}

    for source in args.source:
        if not source.is_dir():
            raise FileNotFoundError(f"Missing source directory: {source}")
        source_report, image_records, crop_records = audit_source(
            source, args.min_crop_size
        )
        sources[source.name] = source_report
        all_image_records.extend(image_records)
        all_crop_records.extend(crop_records)
        preview_sheets[source.name] = []
        for split in SPLITS:
            split_records = [row for row in image_records if row["split"] == split]
            preview_sheets[source.name].extend(
                write_contact_sheets(split_records, args.preview_dir / safe_slug(source.name))
            )

    crop_previews = write_crop_previews(
        all_crop_records, args.preview_dir, args.crop_preview_per_class
    )
    exact_groups, near_pairs, total_near = duplicate_audit(
        all_image_records, args.near_duplicate_threshold
    )
    write_manifest(all_crop_records, args.manifest)

    totals = Counter()
    for crop in all_crop_records:
        totals[crop["label_name"]] += 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "totals": {
            "mapped_crops": dict(sorted(totals.items())),
            "images": len(all_image_records),
            "annotations": sum(
                source["annotations"] for source in sources.values()
            ),
            "exact_duplicate_groups": len(exact_groups),
            "near_duplicate_pairs": total_near,
        },
        "mapping": {
            name: {
                "label": values[0],
                "label_name": values[1],
                "weak_label": values[2],
            }
            for name, values in CATEGORY_RULES.items()
        },
        "unmapped_categories": sorted(UNMAPPED_CATEGORIES),
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_pairs_truncated": total_near > len(near_pairs),
        "crop_previews": crop_previews,
        "annotated_previews": preview_sheets,
        "candidate_manifest": str(args.manifest),
        "limitations": [
            "Crops are from ground-truth COCO boxes, not runtime detector boxes.",
            "The clean-eggs and degrading-quality datasets are web/stock imagery.",
            "`white egg` in the degrading dataset is treated as a weak Clean label.",
            "Stock/watermark detection is filename-based; manual previews show visible watermarks and text overlays that are not counted.",
            "No source/session metadata is present, so leakage cannot be fully ruled out.",
            "This audit does not train a model and does not write crop image files.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"mapped crops: {dict(sorted(totals.items()))}")
    for name, source in sources.items():
        print(
            f"{name}: images={source['images']} annotations={source['annotations']} "
            f"mapped={source['mapped_counts']} invalid_boxes={len(source['invalid_boxes'])}"
        )
    print(f"exact duplicate groups: {len(exact_groups)}")
    print(f"near duplicate pairs: {total_near}")
    print(f"manifest: {args.manifest}")
    print(f"report: {args.report}")
    print(f"previews: {args.preview_dir}")


if __name__ == "__main__":
    main()
