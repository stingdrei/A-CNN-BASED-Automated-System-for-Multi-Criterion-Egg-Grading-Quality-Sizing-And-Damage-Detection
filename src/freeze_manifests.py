"""Freeze and verify split manifests for reproducible evaluation.

Writes a report with per-manifest hashes, counts, cross-split filename/source
overlaps, and optional missing-file checks. This is a no-training release
evidence step.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    if "filename" not in rows[0]:
        raise ValueError(f"Manifest missing filename column: {path}")
    return rows


def git_metadata(root: Path) -> Optional[dict]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": bool(status.strip())}


def cross_split_overlaps(
    manifests: Dict[str, List[dict]], column: str
) -> List[dict]:
    if not all(any(column in row for row in rows) for rows in manifests.values()):
        return []
    owners: Dict[str, List[str]] = defaultdict(list)
    for split, rows in manifests.items():
        for row in rows:
            value = (row.get(column) or "").strip()
            if value:
                owners[value].append(split)
    overlaps = []
    for value, splits in sorted(owners.items()):
        unique_splits = sorted(set(splits))
        if len(unique_splits) > 1:
            overlaps.append({"value": value, "splits": unique_splits})
    return overlaps


def freeze_manifests(
    manifest_paths: Iterable[Path],
    image_root: Path = Path("."),
    check_files: bool = True,
) -> dict:
    paths = list(manifest_paths)
    if not paths:
        raise ValueError("At least one manifest is required")

    names = [path.stem for path in paths]
    if len(names) != len(set(names)):
        raise ValueError(f"Manifest names must be unique: {names}")

    manifests: Dict[str, List[dict]] = {}
    manifest_reports = {}
    for name, path in zip(names, paths):
        rows = read_manifest(path)
        manifests[name] = rows
        missing = []
        if check_files:
            missing = [
                row["filename"]
                for row in rows
                if not (image_root / row["filename"]).is_file()
            ]
        label_counts = Counter(
            (row.get("label_name") or row.get("label") or "unknown")
            for row in rows
        )
        manifest_reports[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "labels": dict(sorted(label_counts.items())),
            "unique_filenames": len({row["filename"] for row in rows}),
            "unique_source_groups": len(
                {row.get("source_group", "") for row in rows if row.get("source_group")}
            ),
            "unique_source_images": len(
                {row.get("source_image", "") for row in rows if row.get("source_image")}
            ),
            "missing_files": missing[:50],
            "missing_file_count": len(missing),
        }

    overlaps = {
        "filenames": cross_split_overlaps(manifests, "filename"),
        "source_groups": cross_split_overlaps(manifests, "source_group"),
        "source_images": cross_split_overlaps(manifests, "source_image"),
        "sha256": cross_split_overlaps(manifests, "sha256"),
    }
    combined = hashlib.sha256(
        "".join(report["sha256"] for report in manifest_reports.values()).encode()
    ).hexdigest()
    overlap_count = sum(len(values) for values in overlaps.values())
    missing_count = sum(
        report["missing_file_count"] for report in manifest_reports.values()
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(Path.cwd()),
        "combined_sha256": combined,
        "manifests": manifest_reports,
        "overlaps": overlaps,
        "overlap_count": overlap_count,
        "missing_file_count": missing_count,
        "release_ready": overlap_count == 0 and missing_count == 0,
        "limitations": [
            "Manifest hashes freeze the split definitions, not model behavior.",
            "Cross-split checks only run for columns present in every manifest.",
            "This does not prove physical-source independence without session metadata.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Manifest CSV; repeat for each split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/manifest_freeze.json"),
    )
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--skip-file-check", action="store_true")
    args = parser.parse_args()

    report = freeze_manifests(
        args.manifest,
        image_root=args.image_root,
        check_files=not args.skip_file_check,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"combined sha256: {report['combined_sha256']}")
    for name, manifest in report["manifests"].items():
        print(
            f"{name}: rows={manifest['rows']} "
            f"missing={manifest['missing_file_count']} "
            f"sha256={manifest['sha256']}"
        )
    print(f"overlaps: {report['overlap_count']}")
    print(f"release_ready: {report['release_ready']}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
