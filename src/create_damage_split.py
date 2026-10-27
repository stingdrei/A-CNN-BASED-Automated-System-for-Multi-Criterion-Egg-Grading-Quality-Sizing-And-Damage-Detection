"""Create stratified damage-classification manifests from class folders.

The v2 manifests include every supported image, remove exact cross-label
duplicates, and record a best-effort source group. Source-level separation is
not guaranteed because the repository does not contain egg/session metadata.

Example:
    python src/create_damage_split.py \
      --source "data/damage" \
      --output data/damage/manifests_v2 \
      --report reports/damage_dataset_v2_report.json
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_LABELS = {"Damaged": 0, "Not Damaged": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_SPLIT_NOTE = (
    "Images are split at image level after exact duplicate removal. The source "
    "folder only exposes legacy damaged, legacy not-damaged, and one broken-egg "
    "capture session, so strict source/session isolation cannot be guaranteed."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_source_id(path: Path, class_name: str) -> str:
    stem = path.stem
    if stem.startswith("brokeneggs_"):
        match = re.match(r"(brokeneggs_\d{4}-\d{2}-\d{2})", stem)
        if match:
            return match.group(1)
    if stem.startswith("damaged_"):
        return "legacy_damaged"
    if stem.startswith("not_damaged_"):
        return "legacy_not_damaged"
    return f"legacy_{class_name.lower().replace(' ', '_')}"


def collect(source: Path) -> pd.DataFrame:
    rows = []
    for class_name, label in CLASS_LABELS.items():
        class_dir = source / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append(
                {
                    "filename": str(path),
                    "label": label,
                    "source_id": infer_source_id(path, class_name),
                    "sha256": file_sha256(path),
                }
            )
    if not rows:
        raise ValueError(f"No supported images found below {source}")
    return pd.DataFrame(rows)


def remove_duplicates(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    keep_indices = []
    conflicts = []
    same_label = []
    for digest, group in data.groupby("sha256", sort=True):
        labels = sorted({int(value) for value in group["label"].tolist()})
        paths = group["filename"].tolist()
        record = {"sha256": digest, "paths": paths, "labels": labels}
        if len(labels) > 1:
            conflicts.append(record)
            continue
        keep_indices.append(group.index[0])
        if len(group) > 1:
            same_label.append(record)
    cleaned = data.loc[keep_indices].copy()
    return cleaned, conflicts, same_label


def split(
    data: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remainder = train_test_split(
        data,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=data["label"],
    )
    relative_test = test_size / (val_size + test_size)
    val, test = train_test_split(
        remainder,
        test_size=relative_test,
        random_state=seed,
        stratify=remainder["label"],
    )
    return train, val, test


def split_summary(frame: pd.DataFrame) -> dict:
    return {
        "images": int(len(frame)),
        "damaged": int((frame["label"] == 0).sum()),
        "not_damaged": int((frame["label"] == 1).sum()),
        "source_counts": {
            str(key): int(value)
            for key, value in frame["source_id"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/damage"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/damage/manifests_v2")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/damage_dataset_v2_report.json"),
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.val_size <= 0
        or args.test_size <= 0
        or args.val_size + args.test_size >= 1
    ):
        raise ValueError(
            "val-size and test-size must be positive and sum to less than 1"
        )

    found = collect(args.source)
    data, conflicts, same_label = remove_duplicates(found)
    train, val, test = split(data, args.val_size, args.test_size, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.sample(frac=1, random_state=args.seed).to_csv(
            args.output / f"{name}.csv", index=False
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "output": str(args.output),
        "images_found": int(len(found)),
        "images_used": int(len(data)),
        "duplicates_removed": int(len(found) - len(data)),
        "conflicting_duplicate_groups": conflicts,
        "same_label_duplicate_groups": same_label,
        "source_groups_found": {
            str(key): int(value)
            for key, value in found["source_id"].value_counts().sort_index().items()
        },
        "strict_source_split": False,
        "split_strategy": "stratified_image_level_exact_duplicates_removed",
        "source_split_note": SOURCE_SPLIT_NOTE,
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "splits": {
            "train": split_summary(train),
            "val": split_summary(val),
            "test": split_summary(test),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"found: {len(found)} images")
    print(f"used: {len(data)} images")
    print(
        "removed: "
        f"{len(conflicts)} conflicting duplicate group(s), "
        f"{len(same_label)} same-label duplicate group(s)"
    )
    for name, frame in (("train", train), ("val", val), ("test", test)):
        summary = split_summary(frame)
        print(
            f"{name}: {summary['images']} images "
            f"(damaged={summary['damaged']}, "
            f"not_damaged={summary['not_damaged']})"
        )
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
