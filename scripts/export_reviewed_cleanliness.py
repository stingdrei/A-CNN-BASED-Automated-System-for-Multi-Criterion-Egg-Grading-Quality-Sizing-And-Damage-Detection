"""Export frozen reviewed-crop manifest for src/train_classifier.py."""
import argparse
import csv
from pathlib import Path


LABEL_IDS = {"Clean": 0, "Stained": 1}
FIELDS = [
    "filename", "label", "label_name", "split", "source_file", "source_group",
    "source_label", "crop_file", "crop_sha256", "crop_width", "crop_height",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Frozen accepted_manifest.csv from training preparation")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partitions = {"train": [], "validation": []}
    seen_groups = {"train": set(), "validation": set()}
    with args.source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in partitions:
                continue
            if row.get("training_eligible", "").lower() != "true":
                raise ValueError(f"Non-eligible crop included in {split}: {row['crop_file']}")
            if row.get("file_exists", "").lower() != "true" or row.get("image_readable", "").lower() != "true":
                raise ValueError(f"Crop failed integrity checks: {row['crop_file']}")
            if row.get("dimension_match", "").lower() != "true":
                raise ValueError(f"Crop dimensions mismatch: {row['crop_file']}")
            if row["reviewed_label"] not in LABEL_IDS:
                raise ValueError(f"Invalid reviewed label: {row['reviewed_label']}")
            group = row["source_group"].strip()
            if not group:
                raise ValueError(f"Missing source group: {row['crop_file']}")
            seen_groups[split].add(group)
            partitions[split].append({
                "filename": f"crops/{row['crop_file']}",
                "label": LABEL_IDS[row["reviewed_label"]],
                "label_name": row["reviewed_label"],
                "split": "valid" if split == "validation" else "train",
                "source_file": row["source_file"],
                "source_group": group,
                "source_label": row["source_label"],
                "crop_file": row["crop_file"],
                "crop_sha256": row["crop_sha256"],
                "crop_width": row["crop_width"],
                "crop_height": row["crop_height"],
            })
    overlap = seen_groups["train"] & seen_groups["validation"]
    if overlap:
        raise ValueError(f"Source groups cross train/validation: {sorted(overlap)}")
    for split, name in (("train", "train.csv"), ("validation", "valid.csv")):
        if not partitions[split]:
            raise ValueError(f"Empty {split} partition")
        if {row["label_name"] for row in partitions[split]} != set(LABEL_IDS):
            raise ValueError(f"{split} partition must contain both classes")
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(partitions[split])
        print(f"{name}: {len(partitions[split])} crops, {len(seen_groups[split])} source groups")
    print("Class mapping: Clean=0, Stained=1")


if __name__ == "__main__":
    main()
