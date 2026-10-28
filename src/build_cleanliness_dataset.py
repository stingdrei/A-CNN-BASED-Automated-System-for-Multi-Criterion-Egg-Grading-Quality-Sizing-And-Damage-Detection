"""Build filtered cleanliness crops with source-image dedupe and splits.

Consumes the candidate manifest from ``src/audit_cleanliness_sources.py`` and
writes actual crop images, source-image-grouped train/val/test manifests, and
an audit report.

This script does not train a model.

Example:
    python src/build_cleanliness_dataset.py \
      --manifest reports/cleanliness_candidate_crops.csv \
      --output data/cleanliness_candidates \
      --report reports/cleanliness_candidates_report.json \
      --exclude-image "Eggs degrading quality/test/7_jpeg...jpg" \
      --exclude-image "Eggs degrading quality/test/images-3-_jpg...jpg"
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
try:
    from src.audit_cleanliness_sources import (
        hamming_distance,
        perceptual_hash,
        safe_slug,
    )
except ImportError:  # Running as ``python src/build_cleanliness_dataset.py``.
    from audit_cleanliness_sources import (
        hamming_distance,
        perceptual_hash,
        safe_slug,
    )

LABEL_NAMES = {"0": "Clean", "1": "Stained"}
SPLITS = ("train", "valid", "test")


def read_manifest(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "source_image",
        "annotation_id",
        "category",
        "label",
        "label_name",
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
    }
    if not rows:
        raise ValueError(f"Empty candidate manifest: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return rows


def image_path(root: Path, row: dict) -> Path:
    path = Path(row["source_image"])
    return path if path.is_absolute() else root / path


def source_image_metadata(
    rows: List[dict], root: Path
) -> Tuple[Dict[str, dict], List[dict]]:
    metadata: Dict[str, dict] = {}
    errors: List[dict] = []
    for row in rows:
        source = row["source_image"]
        if source in metadata:
            continue
        path = image_path(root, row)
        if not path.is_file():
            errors.append({"source_image": source, "error": "missing"})
            continue
        try:
            with Image.open(path) as decoded:
                image = decoded.convert("RGB")
                phash = perceptual_hash(image)
                width, height = image.size
        except (OSError, ValueError) as exc:
            errors.append({"source_image": source, "error": str(exc)})
            continue
        metadata[source] = {
            "path": source,
            "sha256": row["sha256"],
            "phash": phash,
            "width": width,
            "height": height,
        }
    return metadata, errors


def cluster_images(
    metadata: Dict[str, dict], near_duplicate_threshold: int
) -> Tuple[Dict[str, str], List[dict]]:
    """Union source images that are exact or perceptual duplicates."""
    parent = {source: source for source in metadata}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_hash: Dict[str, List[str]] = defaultdict(list)
    for source, info in metadata.items():
        by_hash[info["sha256"]].append(source)
    for sources in by_hash.values():
        for source in sources[1:]:
            union(sources[0], source)

    sources = sorted(metadata)
    for index, left in enumerate(sources):
        for right in sources[index + 1 :]:
            if metadata[left]["sha256"] == metadata[right]["sha256"]:
                continue
            distance = hamming_distance(
                metadata[left]["phash"], metadata[right]["phash"]
            )
            if distance <= near_duplicate_threshold:
                union(left, right)

    groups: Dict[str, List[str]] = defaultdict(list)
    for source in sources:
        groups[find(source)].append(source)

    cluster_ids: Dict[str, str] = {}
    cluster_report = []
    for cluster_index, group in enumerate(sorted(groups.values()), start=1):
        cluster_id = f"source_{cluster_index:04d}"
        for source in group:
            cluster_ids[source] = cluster_id
        cluster_report.append(
            {
                "cluster_id": cluster_id,
                "images": group,
                "image_count": len(group),
            }
        )
    return cluster_ids, cluster_report


def filter_rows(
    rows: List[dict],
    min_crop_size: int,
    include_weak: bool,
    include_stock_keyword: bool,
    exclude_images: set,
) -> Tuple[List[dict], Counter]:
    kept = []
    dropped = Counter()
    for row in rows:
        if row["source_image"] in exclude_images:
            dropped["excluded_image"] += 1
            continue
        if row.get("stock_keyword") and not include_stock_keyword:
            dropped["stock_keyword"] += 1
            continue
        if row.get("weak_label") == "1" and not include_weak:
            dropped["weak_label"] += 1
            continue
        if int(row["crop_width"]) < min_crop_size or int(row["crop_height"]) < min_crop_size:
            dropped["small_crop"] += 1
            continue
        kept.append(row)
    return kept, dropped


def cluster_rows(
    rows: List[dict],
    cluster_ids: Dict[str, str],
) -> Tuple[List[dict], List[dict], Counter]:
    """Group rows into source clusters; drop ambiguous mixed-label clusters."""
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_image"]].append(row)

    clusters: List[dict] = []
    dropped = Counter()
    ambiguous = []
    for source, source_rows in by_source.items():
        if source not in cluster_ids:
            dropped["missing_image"] += len(source_rows)
            continue
        labels = {row["label_name"] for row in source_rows}
        if len(labels) != 1:
            ambiguous.append({"source_image": source, "labels": sorted(labels)})
            dropped["ambiguous_label"] += len(source_rows)
            continue
        clusters.append(
            {
                "cluster_id": cluster_ids[source],
                "source_image": source,
                "label": source_rows[0]["label"],
                "label_name": source_rows[0]["label_name"],
                "rows": source_rows,
            }
        )
    return clusters, ambiguous, dropped


def split_clusters(
    clusters: List[dict], val_size: float, test_size: float, seed: int
) -> Dict[str, List[dict]]:
    """Greedy group split that balances crop counts per label.

    Splitting by source cluster prevents crops from one image leaking across
    splits. Greedy assignment keeps the per-class crop counts close to the
    requested ratios even when source clusters have very different crop counts.
    """
    if len(clusters) < 3:
        raise ValueError("Need at least three source groups to split")
    labels = sorted({cluster["label"] for cluster in clusters})
    if len(labels) != 2:
        raise ValueError("Both Clean and Stained source groups are required")

    ratios = {
        "train": 1.0 - val_size - test_size,
        "valid": val_size,
        "test": test_size,
    }
    rng = random.Random(seed)
    assignments: Dict[str, str] = {}
    for label in labels:
        label_clusters = [c for c in clusters if c["label"] == label]
        if len(label_clusters) < 3:
            raise ValueError(f"Need at least three source groups for label {label}")
        rng.shuffle(label_clusters)
        label_clusters.sort(key=lambda cluster: -len(cluster["rows"]))
        total = sum(len(cluster["rows"]) for cluster in label_clusters)
        targets = {split: total * ratio for split, ratio in ratios.items()}
        assigned = {split: 0 for split in ratios}
        for cluster in label_clusters:
            split = max(ratios, key=lambda name: targets[name] - assigned[name])
            assignments[cluster["cluster_id"]] = split
            assigned[split] += len(cluster["rows"])

    result = {split: [] for split in ratios}
    for cluster in clusters:
        result[assignments[cluster["cluster_id"]]].append(cluster)

    for split, split_clusters_list in result.items():
        split_labels = {cluster["label"] for cluster in split_clusters_list}
        if split_labels != set(labels):
            raise ValueError(f"Split {split} is missing a class: {sorted(split_labels)}")
    return result


def crop_with_padding(image: Image.Image, row: dict, padding: float) -> Image.Image:
    width, height = image.size
    x1 = max(0, int(row["bbox_x1"]) - int(width * padding))
    y1 = max(0, int(row["bbox_y1"]) - int(height * padding))
    x2 = min(width, int(row["bbox_x2"]) + int(width * padding))
    y2 = min(height, int(row["bbox_y2"]) + int(height * padding))
    return image.crop((x1, y1, x2, y2))


def write_crops(
    split_clusters_map: Dict[str, List[dict]],
    output: Path,
    image_root: Path,
    padding: float,
) -> Tuple[List[dict], List[dict]]:
    records: List[dict] = []
    errors: List[dict] = []
    for split, clusters in split_clusters_map.items():
        for cluster in clusters:
            source_path = image_path(image_root, cluster["rows"][0])
            try:
                with Image.open(source_path) as decoded:
                    image = decoded.convert("RGB")
            except OSError as exc:
                errors.append({"source_image": cluster["source_image"], "error": str(exc)})
                continue

            for row in cluster["rows"]:
                label_name = row["label_name"]
                output_dir = output / "crops" / split / label_name
                output_dir.mkdir(parents=True, exist_ok=True)
                crop = crop_with_padding(image, row, padding)
                if crop.width < 2 or crop.height < 2:
                    errors.append(
                        {"source_image": cluster["source_image"], "error": "empty crop"}
                    )
                    continue
                name = (
                    f"{safe_slug(Path(cluster['source_image']).stem)[:90]}"
                    f"__egg{int(row['annotation_id']):05d}.jpg"
                )
                output_path = output_dir / name
                crop.save(output_path, quality=95)
                records.append(
                    {
                        "filename": output_path.as_posix(),
                        "label": row["label"],
                        "label_name": label_name,
                        "split": split,
                        "source_image": cluster["source_image"],
                        "source_group": cluster["cluster_id"],
                        "source_dataset": row["dataset"],
                        "category": row["category"],
                        "weak_label": row["weak_label"],
                        "bbox_x1": row["bbox_x1"],
                        "bbox_y1": row["bbox_y1"],
                        "bbox_x2": row["bbox_x2"],
                        "bbox_y2": row["bbox_y2"],
                        "crop_width": str(crop.width),
                        "crop_height": str(crop.height),
                        "image_width": row["image_width"],
                        "image_height": row["image_height"],
                        "sha256": row["sha256"],
                    }
                )
    return records, errors


def write_manifest(records: List[dict], path: Path) -> None:
    fields = [
        "filename",
        "label",
        "label_name",
        "split",
        "source_image",
        "source_group",
        "source_dataset",
        "category",
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_readme(output: Path) -> None:
    content = """# Cleanliness candidate crops

Generated from the web/stock Roboflow sources `clean eggs/` and
`Eggs degrading quality/`. This is weak pretraining data, not a release test
set. It contains background/domain bias and possible visible watermarks.

Mapping:
- `Clean`: Brownegg, Lightbrownegg, Whiteegg from `clean eggs/`
- `Stained`: dirt stained brown/white egg from `Eggs degrading quality/`
- Excluded: weak `white egg` labels, stock-keyword images, known text overlays,
  invalid/small boxes, and near-duplicate source images.

Splits are assigned by source image cluster, so all crops from one source image
stay in one split. Source datasets have no session IDs; this is the strongest
leakage control available.

Train only after manually reviewing the previews in
`reports/cleanliness_source_previews/`.
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/cleanliness_candidate_crops.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cleanliness_candidates"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/cleanliness_candidates_report.json"),
    )
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--min-crop-size", type=int, default=24)
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-weak", action="store_true")
    parser.add_argument("--include-stock-keyword", action="store_true")
    parser.add_argument(
        "--exclude-image",
        action="append",
        default=[],
        help="Source image path to exclude; repeat for multiple images",
    )
    args = parser.parse_args()

    if args.min_crop_size < 1:
        raise ValueError("--min-crop-size must be positive")
    if args.val_size <= 0 or args.test_size <= 0 or args.val_size + args.test_size >= 1:
        raise ValueError("--val-size and --test-size must sum to less than 1")
    if args.padding < 0:
        raise ValueError("--padding must be non-negative")

    rows = read_manifest(args.manifest)
    metadata, image_errors = source_image_metadata(rows, args.image_root)
    cluster_ids, cluster_report = cluster_images(
        metadata, args.near_duplicate_threshold
    )
    keep_rows, dropped = filter_rows(
        rows,
        args.min_crop_size,
        args.include_weak,
        args.include_stock_keyword,
        set(args.exclude_image),
    )
    clusters, ambiguous, cluster_dropped = cluster_rows(keep_rows, cluster_ids)
    dropped.update(cluster_dropped)

    # Keep one representative source image per duplicate cluster.
    by_cluster: Dict[str, List[dict]] = defaultdict(list)
    for cluster in clusters:
        by_cluster[cluster["cluster_id"]].append(cluster)
    representative_clusters = []
    for cluster_id, candidates in by_cluster.items():
        candidates.sort(key=lambda item: (-len(item["rows"]), item["source_image"]))
        representative = candidates[0]
        dropped["duplicate_source_crops"] += sum(
            len(candidate["rows"]) for candidate in candidates[1:]
        )
        representative_clusters.append(representative)
        for candidate in candidates[1:]:
            cluster_report.append(
                {
                    "cluster_id": cluster_id,
                    "representative": representative["source_image"],
                    "dropped_duplicate": candidate["source_image"],
                    "label_name": candidate["label_name"],
                    "dropped_crops": len(candidate["rows"]),
                }
            )

    split_map = split_clusters(
        representative_clusters, args.val_size, args.test_size, args.seed
    )
    records, crop_errors = write_crops(
        split_map, args.output, args.image_root, args.padding
    )
    for split, clusters_in_split in split_map.items():
        split_records = [row for row in records if row["split"] == split]
        write_manifest(split_records, args.output / "manifests" / f"{split}.csv")
    write_readme(args.output)

    split_summary = {}
    for split, clusters_in_split in split_map.items():
        split_records = [row for row in records if row["split"] == split]
        split_summary[split] = {
            "crops": len(split_records),
            "by_label": dict(
                sorted(Counter(row["label_name"] for row in split_records).items())
            ),
            "source_groups": len(clusters_in_split),
            "source_images": len({row["source_image"] for row in split_records}),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest": str(args.manifest),
        "output": str(args.output),
        "filters": {
            "min_crop_size": args.min_crop_size,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "padding": args.padding,
            "include_weak": args.include_weak,
            "include_stock_keyword": args.include_stock_keyword,
            "exclude_images": args.exclude_image,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "seed": args.seed,
        },
        "input_crops": len(rows),
        "kept_crops": len(records),
        "dropped": dict(sorted(dropped.items())),
        "splits": split_summary,
        "duplicate_clusters": cluster_report,
        "ambiguous_clusters": ambiguous,
        "image_errors": image_errors,
        "crop_errors": crop_errors,
        "limitations": [
            "Web/stock domain, not physical tray images.",
            "Known stock-keyword and manual exclusions only; some watermarks remain.",
            "No session IDs; splits are grouped by source image/duplicate cluster only.",
            "This dataset is for initial training, not release validation.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"input crops: {len(rows)}")
    print(f"kept crops: {len(records)}")
    print(f"dropped: {dict(sorted(dropped.items()))}")
    for split, summary in split_summary.items():
        print(f"{split}: {summary}")
    print(f"output: {args.output}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
