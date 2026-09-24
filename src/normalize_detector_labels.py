"""Normalize detector annotations to the single ``egg`` class.

The detector locates eggs. Damage is a downstream classification task and
must not be represented as detector classes.

Usage:
    python src/normalize_detector_labels.py --in-place
"""

import argparse
from pathlib import Path


def normalize(labels_root: Path, in_place: bool) -> int:
    files = sorted(labels_root.glob("*/**/*.txt"))
    if not files:
        raise FileNotFoundError(f"No YOLO label files found under {labels_root}")

    changed = 0
    for path in files:
        lines = path.read_text().splitlines()
        normalized = []
        for line_number, line in enumerate(lines, 1):
            fields = line.split()
            if not line.strip():
                continue
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
            normalized.append("0 " + " ".join(fields[1:]))
        content = "\n".join(normalized) + ("\n" if normalized else "")
        if in_place:
            if path.read_text() != content:
                path.write_text(content)
                changed += 1
        else:
            print(f"{path}: {len(normalized)} egg annotation(s)")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", type=Path, default=Path("data/detection/labels"))
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    changed = normalize(args.labels_root, args.in_place)
    if args.in_place:
        print(f"Normalized {changed} label file(s) to class 0 (egg).")


if __name__ == "__main__":
    main()
