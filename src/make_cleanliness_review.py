"""Generate cleanliness crop review sheets and an editable review CSV.

The CSV is prefilled with ``decision=keep``. Change rows to ``exclude`` for
watermarked, ambiguous, mislabeled, or otherwise unusable crops, then run
``src/apply_cleanliness_review.py``.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


DEFAULT_REVIEW_ID = "keep"
EXCLUDE_DECISIONS = {"exclude", "reject", "bad", "drop"}


def read_manifest_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "filename" not in rows[0]:
        raise ValueError(f"Manifest missing filename column: {path}")
    return rows


def load_existing_decisions(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            row["filename"]: {
                "decision": (row.get("decision") or DEFAULT_REVIEW_ID).strip(),
                "notes": (row.get("notes") or "").strip(),
            }
            for row in rows
            if row.get("filename")
        }


def make_review_rows(
    manifests: dict[str, list[dict]], existing: dict
) -> list[dict]:
    rows = []
    for split, manifest_rows in manifests.items():
        ordered = sorted(
            manifest_rows,
            key=lambda row: (row.get("source_group", ""), row["filename"]),
        )
        for index, row in enumerate(ordered, start=1):
            review_id = f"{split}_{row.get('label_name', 'unknown')}_{index:04d}"
            previous = existing.get(row["filename"], {})
            rows.append(
                {
                    "review_id": review_id,
                    "split": split,
                    "label_name": row.get("label_name", ""),
                    "filename": row["filename"],
                    "source_group": row.get("source_group", ""),
                    "source_image": row.get("source_image", ""),
                    "decision": previous.get("decision", DEFAULT_REVIEW_ID),
                    "notes": previous.get("notes", ""),
                }
            )
    return rows


def write_review_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "review_id",
        "split",
        "label_name",
        "filename",
        "source_group",
        "source_image",
        "decision",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheets(
    rows: list[dict],
    image_root: Path,
    output_dir: Path,
    thumb_size: int,
    per_sheet: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["split"], row["label_name"])].append(row)

    cells = max(1, int(per_sheet))
    columns = min(8, cells)
    lines = max(1, (cells + columns - 1) // columns)
    cell_height = thumb_size + 20
    sheets = []

    for (split, label_name), group_rows in sorted(by_group.items()):
        for sheet_index in range(0, len(group_rows), cells):
            chunk = group_rows[sheet_index : sheet_index + cells]
            canvas = Image.new(
                "RGB",
                (columns * thumb_size, lines * cell_height),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            for offset, row in enumerate(chunk):
                image_path = image_root / row["filename"]
                try:
                    with Image.open(image_path) as decoded:
                        image = ImageOps.exif_transpose(decoded).convert("RGB")
                        image.thumbnail((thumb_size, thumb_size))
                except OSError:
                    image = Image.new("RGB", (thumb_size, thumb_size), (230, 230, 230))
                column = offset % columns
                line = offset // columns
                left = column * thumb_size
                top = line * cell_height
                x = left + (thumb_size - image.width) // 2
                y = top + 20 + (thumb_size - image.height) // 2
                canvas.paste(image, (x, y))
                draw.text((left + 2, top + 2), row["review_id"], fill="red")
            sheet_name = f"{split}_{label_name}_{sheet_index // cells:02d}.jpg"
            sheet_path = output_dir / sheet_name
            canvas.save(sheet_path, quality=90)
            sheets.append(str(sheet_path))
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/cleanliness_candidates/manifests"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/cleanliness_review"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("reports/cleanliness_review.csv"),
    )
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--thumb-size", type=int, default=160)
    parser.add_argument("--per-sheet", type=int, default=64)
    args = parser.parse_args()

    manifest_paths = {
        path.stem: path
        for path in sorted(args.manifest_dir.glob("*.csv"))
        if path.stem in {"train", "valid", "test"}
    }
    if not manifest_paths:
        raise FileNotFoundError(f"No train/valid/test manifests in {args.manifest_dir}")
    if args.thumb_size < 32:
        raise ValueError("--thumb-size must be at least 32")
    if args.per_sheet < 1:
        raise ValueError("--per-sheet must be positive")

    manifests = {split: read_manifest_rows(path) for split, path in manifest_paths.items()}
    existing = load_existing_decisions(args.csv)
    rows = make_review_rows(manifests, existing)
    write_review_csv(rows, args.csv)
    sheets = make_contact_sheets(
        rows,
        args.image_root,
        args.output_dir,
        args.thumb_size,
        args.per_sheet,
    )
    print(f"review rows: {len(rows)}")
    print(f"sheets: {len(sheets)}")
    print(f"csv: {args.csv}")
    print(f"previews: {args.output_dir}")


if __name__ == "__main__":
    main()
