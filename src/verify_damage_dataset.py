"""Verify damage manifests: files, balance, and exact duplicate leakage."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest_dir: Path, image_root: Path) -> tuple[int, dict]:
    errors = 0
    seen_hashes: dict[str, dict] = {}
    duplicates = []
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_dir": str(manifest_dir),
        "image_root": str(image_root),
        "splits": {},
        "exact_duplicate_groups": duplicates,
        "errors": 0,
    }

    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.csv"
        if not path.is_file():
            print(f"{split}: missing manifest {path}")
            errors += 1
            report["splits"][split] = {"error": "missing manifest"}
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"filename", "label"}
        if not rows or not required.issubset(rows[0]):
            print(f"{split}: invalid columns; expected filename,label")
            errors += 1
            report["splits"][split] = {"error": "invalid columns"}
            continue

        missing = []
        labels = {"0": 0, "1": 0}
        source_counts: dict[str, int] = {}
        hashes: dict[str, str] = {}
        for row in rows:
            label = row["label"]
            if label not in labels:
                errors += 1
                continue
            labels[label] += 1
            source_id = row.get("source_id", "unknown")
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            image_path = image_root / row["filename"]
            if not image_path.is_file():
                missing.append(row["filename"])
                continue
            digest = file_sha256(image_path)
            hashes[row["filename"]] = digest
            previous = seen_hashes.get(digest)
            if previous is not None:
                duplicate = {
                    "sha256": digest,
                    "current": {"split": split, "filename": row["filename"]},
                    "previous": previous,
                }
                duplicates.append(duplicate)
                errors += 1
            else:
                seen_hashes[digest] = {
                    "split": split,
                    "filename": row["filename"],
                    "label": label,
                }

        print(
            f"{split}: total={len(rows)} damaged={labels['0']} "
            f"not_damaged={labels['1']} missing={len(missing)} "
            f"duplicate_hashes={len(hashes) - len(set(hashes.values()))}"
        )
        if missing:
            print(f"  first missing file: {missing[0]}")
            errors += 1
        report["splits"][split] = {
            "total": len(rows),
            "damaged": labels["0"],
            "not_damaged": labels["1"],
            "missing": missing,
            "source_counts": dict(sorted(source_counts.items())),
        }

    report["errors"] = errors
    report["exact_duplicate_count"] = len(duplicates)
    return errors, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/damage/manifests")
    )
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    errors, report = verify(args.manifest_dir, args.image_root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    if errors:
        raise SystemExit(f"Dataset verification failed with {errors} issue(s)")
    print("Damage dataset verification passed")


if __name__ == "__main__":
    main()
