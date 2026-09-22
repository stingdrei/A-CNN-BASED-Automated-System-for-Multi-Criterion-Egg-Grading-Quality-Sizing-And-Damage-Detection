"""Verify damage manifests and report class balance and missing files."""

import argparse
import csv
from pathlib import Path


def verify(manifest_dir: Path) -> int:
    errors = 0
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.csv"
        if not path.is_file():
            print(f"{split}: missing manifest {path}")
            errors += 1
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"filename", "label"}
        if not rows or not required.issubset(rows[0]):
            print(f"{split}: invalid columns; expected filename,label")
            errors += 1
            continue
        missing = [row["filename"] for row in rows if not Path(row["filename"]).is_file()]
        labels = {label: sum(row["label"] == label for row in rows) for label in ("0", "1")}
        print(
            f"{split}: total={len(rows)} damaged={labels['0']} "
            f"not_damaged={labels['1']} missing={len(missing)}"
        )
        if missing:
            print(f"  first missing file: {missing[0]}")
            errors += 1
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/damage/manifests"))
    args = parser.parse_args()
    errors = verify(args.manifest_dir)
    if errors:
        raise SystemExit(f"Dataset verification failed with {errors} issue(s)")
    print("Damage dataset verification passed")


if __name__ == "__main__":
    main()
