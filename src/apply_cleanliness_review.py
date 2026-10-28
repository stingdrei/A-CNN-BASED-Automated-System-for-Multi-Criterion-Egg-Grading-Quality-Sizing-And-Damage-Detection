"""Apply manual cleanliness review decisions to the candidate crop dataset.

Rows marked ``exclude`` (or ``reject``/``bad``/``drop``) are removed from the
split manifests and moved to an excluded directory. Review decisions are
preserved in ``reports/cleanliness_review.csv``.
"""

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXCLUDE_DECISIONS = {"exclude", "reject", "bad", "drop"}


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}__excluded{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique destination for {path}")


def apply_review(
    manifest_dir: Path,
    review_csv: Path,
    image_root: Path = Path("."),
    excluded_dir: Path = Path("data/cleanliness_candidates/excluded"),
    dry_run: bool = False,
) -> dict:
    review_fields, review_rows = read_csv(review_csv)
    if "filename" not in review_fields or "decision" not in review_fields:
        raise ValueError("Review CSV needs filename and decision columns")

    excluded = {
        row["filename"]
        for row in review_rows
        if (row.get("decision") or "").strip().lower() in EXCLUDE_DECISIONS
    }
    manifest_paths = {
        path.stem: path
        for path in sorted(manifest_dir.glob("*.csv"))
        if path.stem in {"train", "valid", "test"}
    }
    if not manifest_paths:
        raise FileNotFoundError(f"No train/valid/test manifests in {manifest_dir}")

    seen_filenames = set()
    removed_rows = []
    moved_files = []
    missing_files = []
    split_reports = {}
    for split, path in manifest_paths.items():
        fields, rows = read_csv(path)
        if "filename" not in fields:
            raise ValueError(f"Manifest missing filename column: {path}")
        seen_filenames.update(row["filename"] for row in rows)
        kept = []
        for row in rows:
            if row["filename"] in excluded:
                removed_rows.append(row)
                source = image_root / row["filename"]
                label_name = row.get("label_name") or row.get("label") or "unknown"
                destination = unique_destination(
                    excluded_dir / split / label_name / source.name
                )
                if source.is_file():
                    if not dry_run:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(source), str(destination))
                    moved_files.append(
                        {"source": str(source), "destination": str(destination)}
                    )
                else:
                    missing_files.append(str(source))
            else:
                kept.append(row)
        split_reports[split] = {
            "before": len(rows),
            "after": len(kept),
            "removed": len(rows) - len(kept),
        }
        if not dry_run:
            write_csv(path, fields, kept)

    unmatched = sorted(excluded - seen_filenames)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "review_csv": str(review_csv),
        "manifest_dir": str(manifest_dir),
        "excluded_dir": str(excluded_dir),
        "reviewed_rows": len(review_rows),
        "excluded_requests": len(excluded),
        "removed_rows": len(removed_rows),
        "moved_files": len(moved_files),
        "missing_files": missing_files,
        "unmatched_review_filenames": unmatched,
        "splits": split_reports,
        "decisions": dict(
            sorted(
                Counter(
                    (row.get("decision") or "").strip().lower()
                    for row in review_rows
                ).items()
            )
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/cleanliness_candidates/manifests"),
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("reports/cleanliness_review.csv"),
    )
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument(
        "--excluded-dir",
        type=Path,
        default=Path("data/cleanliness_candidates/excluded"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/cleanliness_review_applied.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = apply_review(
        args.manifest_dir,
        args.review_csv,
        image_root=args.image_root,
        excluded_dir=args.excluded_dir,
        dry_run=args.dry_run,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"excluded requests: {report['excluded_requests']}")
    print(f"removed rows: {report['removed_rows']}")
    print(f"moved files: {report['moved_files']}")
    for split, summary in report["splits"].items():
        print(f"{split}: {summary}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
