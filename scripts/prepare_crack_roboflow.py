"""Build crack-only binary manifests with source-grouped 80/10/10 splits."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold


CLASS_NAMES = {0: "Damaged", 1: "Not Damaged"}
LABEL_COLUMNS = ("clean", "crack", "dirty")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_group(filename: str) -> str:
    """Remove Roboflow's generated variant suffix, retaining source stem."""
    return re.sub(r"\.rf\.[^.]+(?=\.[^.]+$)", "", Path(filename).name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Roboflow export root containing train/_classes.csv")
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory for manifests and audit JSON")
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    source = args.source.resolve()
    labels_csv = source / "train" / "_classes.csv"
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Expected Roboflow labels file: {labels_csv}")

    rows = []
    missing = []
    with labels_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"filename", *LABEL_COLUMNS}.issubset(reader.fieldnames):
            raise ValueError(f"Expected filename,{','.join(LABEL_COLUMNS)} columns")
        for row in reader:
            filename = row["filename"].strip()
            values = {name: int(row[name]) for name in LABEL_COLUMNS}
            if any(value not in (0, 1) for value in values.values()):
                raise ValueError(f"Non-binary labels for {filename}: {values}")
            if values["crack"]:
                label = 0
            else:
                # Both clean+dirty and single clean/dirty labels mean intact shell
                # for this crack-only task. Cleanliness is not modeled here.
                label = 1
            image_path = source / "train" / filename
            if not image_path.is_file():
                missing.append(filename)
                continue
            rows.append({
                "filename": f"train/{filename}",
                "label": label,
                "label_name": CLASS_NAMES[label],
                "source_group": source_group(filename),
                "sha256": sha256(image_path),
                "original_clean": values["clean"],
                "original_crack": values["crack"],
                "original_dirty": values["dirty"],
            })
    if missing:
        raise FileNotFoundError(f"{len(missing)} labeled images missing; examples: {missing[:5]}")
    if not rows:
        raise ValueError("No labeled images found")

    images = {p.name for p in (source / "train").iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    listed = {Path(row["filename"]).name for row in rows}
    unlisted = sorted(images - listed)
    if unlisted:
        raise ValueError(f"{len(unlisted)} images lack CSV labels; examples: {unlisted[:5]}")

    labels = [row["label"] for row in rows]
    groups = [row["source_group"] for row in rows]
    # Ten approximately equal group-stratified folds; reserve distinct folds for
    # validation and test, with remaining eight folds for training.
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_for_index = {}
    for fold, (_, held_indices) in enumerate(splitter.split(rows, labels, groups)):
        for index in held_indices:
            fold_for_index[index] = fold
    if len(fold_for_index) != len(rows):
        raise RuntimeError("Grouped split did not assign every image")

    split_names = {0: "test", 1: "validation"}
    split_rows = {"train": [], "validation": [], "test": []}
    split_groups = {name: set() for name in split_rows}
    for index, row in enumerate(rows):
        split = split_names.get(fold_for_index[index], "train")
        split_rows[split].append(row)
        split_groups[split].add(row["source_group"])
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_groups[a] & split_groups[b]
        if overlap:
            raise RuntimeError(f"Source group leakage between {a}/{b}: {sorted(overlap)[:5]}")

    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "label", "label_name", "source_group", "sha256",
              "original_clean", "original_crack", "original_dirty"]
    for split, split_data in split_rows.items():
        with (args.output / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(split_data)

    counts = {
        split: {
            "images": len(items),
            "source_groups": len(split_groups[split]),
            "labels": dict(Counter(row["label_name"] for row in items)),
        }
        for split, items in split_rows.items()
    }
    duplicate_hashes = defaultdict(list)
    for row in rows:
        duplicate_hashes[row["sha256"]].append(row["filename"])
    duplicate_groups = [files for files in duplicate_hashes.values() if len(files) > 1]
    audit = {
        "task": "binary crack damage classification",
        "definition": "Damaged=crack label; Not Damaged=clean or dirty without crack",
        "source": str(source),
        "source_labels_sha256": sha256(labels_csv),
        "seed": args.seed,
        "split_method": "StratifiedGroupKFold, 10 folds; fold 0=test, fold 1=validation, folds 2-9=train",
        "total_images": len(rows),
        "total_source_groups": len(set(groups)),
        "split_counts": counts,
        "source_group_overlap": 0,
        "exact_duplicate_sha256_groups": duplicate_groups,
        "unlisted_image_count": len(unlisted),
        "note": "Roboflow export provided no validation/test directories. Splits created here; source-group proxy strips .rf.<hash> suffix. Test remains same-source-dataset evaluation, not independent camera test.",
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
