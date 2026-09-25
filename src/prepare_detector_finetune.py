"""Create a one-class YOLO dataset for detector fine-tuning.

The source dataset is copied, not modified. All existing detector classes are
mapped to class 0 (``egg``), while images and train/val/test membership remain
unchanged.
"""

import argparse
import shutil
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def resolve_root(config_path: Path, config: dict) -> Path:
    configured = Path(str(config.get("path", ".")))
    if configured.is_absolute():
        resolved = configured.resolve()
    else:
        resolved = (config_path.parent / configured).resolve()
    if (resolved / "images").is_dir() and (resolved / "labels").is_dir():
        return resolved
    local_root = config_path.parent.resolve()
    if (local_root / "images").is_dir() and (local_root / "labels").is_dir():
        return local_root
    return resolved


def copy_dataset(source_yaml: Path, output_root: Path) -> None:
    source_yaml = source_yaml.resolve()
    config = yaml.safe_load(source_yaml.read_text()) or {}
    source_root = resolve_root(source_yaml, config)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source_root}")

    output_root = output_root.resolve()
    if output_root == source_root:
        raise ValueError("Output dataset must differ from the source dataset")
    output_root.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0
    for split in SPLITS:
        source_images = source_root / "images" / split
        source_labels = source_root / "labels" / split
        if not source_images.is_dir():
            raise FileNotFoundError(f"Missing source image split: {source_images}")
        if not source_labels.is_dir():
            raise FileNotFoundError(f"Missing source label split: {source_labels}")

        target_images = output_root / "images" / split
        target_labels = output_root / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(source_images.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                shutil.copy2(image_path, target_images / image_path.name)
                copied_images += 1

        image_stems = {
            path.stem
            for path in target_images.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        for label_path in sorted(source_labels.glob("*.txt")):
            if label_path.stem not in image_stems:
                raise ValueError(f"Label has no matching image: {label_path}")
            normalized = []
            for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
                normalized.append("0 " + " ".join(fields[1:]))
            (target_labels / label_path.name).write_text(
                "\n".join(normalized) + ("\n" if normalized else "")
            )
            copied_labels += 1

    output_config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["egg"],
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(output_config, sort_keys=False)
    )
    print(f"Created one-class fine-tuning dataset: {output_root}")
    print(f"Copied {copied_images} images and normalized {copied_labels} label files.")
    print("Original dataset was not modified.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-yaml", type=Path, default=Path("data/detection/data.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/detection_finetune"),
    )
    args = parser.parse_args()
    copy_dataset(args.source_yaml, args.output_root)


if __name__ == "__main__":
    main()
