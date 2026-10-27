"""Build a 1-5 egg smoke dataset from the locked detector test split.

This is not a beta acceptance gate. It filters out large trays so the static
detector demo can be checked on the supported small-tray range.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def box_count(label_path: Path) -> int:
    return sum(
        bool(line.strip()) for line in label_path.read_text().splitlines()
    )


def resolve_root(config_path: Path, config: dict) -> Path:
    root = Path(str(config.get("path", ".")))
    if root.is_absolute():
        return root.resolve()
    return (config_path.parent / root).resolve()


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    os.symlink(source.resolve(), destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-yaml", type=Path, default=Path("data/detection_finetune/data.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/detection_smoke")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/detector_smoke_subset_report.json"),
    )
    parser.add_argument("--min-eggs", type=int, default=1)
    parser.add_argument("--max-eggs", type=int, default=5)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.min_eggs < 0 or args.max_eggs < args.min_eggs:
        raise ValueError("invalid egg range")
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    config = yaml.safe_load(args.data_yaml.read_text())
    source_root = resolve_root(args.data_yaml, config)
    selected = []
    excluded = []
    distribution = Counter()

    for split in SPLITS:
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        for image_path in sorted(image_dir.iterdir()):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_SUFFIXES
            ):
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label: {label_path}")
            count = box_count(label_path)
            if split == "test":
                distribution[count] += 1
                if not (args.min_eggs <= count <= args.max_eggs):
                    excluded.append(
                        {
                            "image": image_path.name,
                            "eggs": count,
                        }
                    )
                    continue
                selected.append(
                    {
                        "image": image_path.name,
                        "label": label_path.name,
                        "eggs": count,
                    }
                )
            link(image_path, args.output / "images" / split / image_path.name)
            link(label_path, args.output / "labels" / split / label_path.name)

    if not selected:
        raise RuntimeError(
            f"No test images in egg range {args.min_eggs}-{args.max_eggs}"
        )

    smoke_config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": int(config["nc"]),
        "names": config["names"],
    }
    (args.output / "data.yaml").write_text(
        yaml.safe_dump(smoke_config, sort_keys=False), encoding="utf-8"
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_yaml": str(args.data_yaml),
        "source_root": str(source_root),
        "output": str(args.output),
        "min_eggs": args.min_eggs,
        "max_eggs": args.max_eggs,
        "test_images_before": sum(distribution.values()),
        "test_images_selected": len(selected),
        "test_images_excluded": len(excluded),
        "test_egg_distribution_before": {
            str(key): value for key, value in sorted(distribution.items())
        },
        "selected_eggs_total": sum(item["eggs"] for item in selected),
        "selected_images": selected,
        "excluded_images": excluded,
        "limitations": [
            "Smoke subset only; not a beta acceptance gate.",
            "Filtering by ground-truth tray count is not possible at inference.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"test images before: {sum(distribution.values())}")
    print(f"selected: {len(selected)}")
    print(f"excluded: {len(excluded)}")
    print(f"distribution: {dict(sorted(distribution.items()))}")
    print(f"smoke yaml: {args.output / 'data.yaml'}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
