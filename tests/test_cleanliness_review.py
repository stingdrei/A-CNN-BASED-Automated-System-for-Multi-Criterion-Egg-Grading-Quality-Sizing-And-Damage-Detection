import csv
from pathlib import Path

from PIL import Image

from src.apply_cleanliness_review import apply_review
from src.make_cleanliness_review import (
    load_existing_decisions,
    make_contact_sheets,
    make_review_rows,
    write_review_csv,
)


def write_manifest(path, rows):
    fields = [
        "filename",
        "label",
        "label_name",
        "split",
        "source_group",
        "source_image",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_rows(tmp_path, split="train", count=2):
    rows = []
    for index in range(count):
        filename = f"data/cleanliness_candidates/crops/{split}/Clean/crop{index}.jpg"
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 48), "white").save(path)
        rows.append(
            {
                "filename": filename,
                "label": "0",
                "label_name": "Clean",
                "split": split,
                "source_group": "source_0001",
                "source_image": f"source{index}.jpg",
            }
        )
    return rows


def test_make_review_rows_preserves_existing_decision(tmp_path):
    rows = make_rows(tmp_path, count=2)
    manifests = {"train": rows}
    review_csv = tmp_path / "review.csv"
    first_rows = make_review_rows(manifests, {})
    first_rows[0]["decision"] = "exclude"
    first_rows[0]["notes"] = "watermark"
    write_review_csv(first_rows, review_csv)

    existing = load_existing_decisions(review_csv)
    second_rows = make_review_rows(manifests, existing)

    assert second_rows[0]["decision"] == "exclude"
    assert second_rows[0]["notes"] == "watermark"
    assert second_rows[1]["decision"] == "keep"


def test_review_decision_is_stable_by_filename_after_reindex(tmp_path):
    rows = make_rows(tmp_path, count=2)
    first_rows = make_review_rows({"train": rows}, {})
    first_rows[1]["decision"] = "exclude"
    first_rows[1]["notes"] = "bad crop"
    review_csv = tmp_path / "review.csv"
    write_review_csv(first_rows, review_csv)

    existing = load_existing_decisions(review_csv)
    shifted_rows = make_review_rows({"train": [rows[1]]}, existing)

    assert shifted_rows[0]["filename"] == rows[1]["filename"]
    assert shifted_rows[0]["review_id"] != first_rows[1]["review_id"]
    assert shifted_rows[0]["decision"] == "exclude"
    assert shifted_rows[0]["notes"] == "bad crop"


def test_contact_sheets_are_generated(tmp_path):
    rows = make_rows(tmp_path, count=2)
    review_rows = make_review_rows({"train": rows}, {})
    sheets = make_contact_sheets(
        review_rows,
        image_root=tmp_path,
        output_dir=tmp_path / "sheets",
        thumb_size=48,
        per_sheet=1,
    )

    assert len(sheets) == 2
    assert all(Path(sheet).is_file() for sheet in sheets)


def test_apply_review_filters_manifest_and_moves_crop(tmp_path):
    rows = make_rows(tmp_path, count=2)
    manifest = tmp_path / "manifests" / "train.csv"
    write_manifest(manifest, rows)
    review_csv = tmp_path / "review.csv"
    review_rows = make_review_rows({"train": rows}, {})
    review_rows[1]["decision"] = "exclude"
    write_review_csv(review_rows, review_csv)

    report = apply_review(
        manifest.parent,
        review_csv,
        image_root=tmp_path,
        excluded_dir=tmp_path / "excluded",
    )

    remaining = list(csv.DictReader(manifest.open(newline="")))
    assert len(remaining) == 1
    assert remaining[0]["filename"] == rows[0]["filename"]
    assert not (tmp_path / rows[1]["filename"]).exists()
    assert (tmp_path / "excluded" / "train" / "Clean" / "crop1.jpg").is_file()
    assert report["removed_rows"] == 1
    assert report["moved_files"] == 1


def test_apply_review_dry_run_keeps_files(tmp_path):
    rows = make_rows(tmp_path, count=1)
    manifest = tmp_path / "manifests" / "train.csv"
    write_manifest(manifest, rows)
    review_csv = tmp_path / "review.csv"
    review_rows = make_review_rows({"train": rows}, {})
    review_rows[0]["decision"] = "exclude"
    write_review_csv(review_rows, review_csv)

    report = apply_review(
        manifest.parent,
        review_csv,
        image_root=tmp_path,
        excluded_dir=tmp_path / "excluded",
        dry_run=True,
    )

    assert report["moved_files"] == 1
    assert (tmp_path / rows[0]["filename"]).is_file()
    assert len(list(csv.DictReader(manifest.open(newline="")))) == 1


def test_apply_review_reports_unmatched_filename(tmp_path):
    rows = make_rows(tmp_path, count=1)
    manifest = tmp_path / "manifests" / "train.csv"
    write_manifest(manifest, rows)
    review_csv = tmp_path / "review.csv"
    review_rows = make_review_rows({"train": rows}, {})
    review_rows.append(
        {
            "review_id": "train_Clean_9999",
            "split": "train",
            "label_name": "Clean",
            "filename": "missing.jpg",
            "source_group": "",
            "source_image": "",
            "decision": "exclude",
            "notes": "",
        }
    )
    write_review_csv(review_rows, review_csv)

    report = apply_review(
        manifest.parent,
        review_csv,
        image_root=tmp_path,
        excluded_dir=tmp_path / "excluded",
    )

    assert report["unmatched_review_filenames"] == ["missing.jpg"]
    assert report["missing_files"] == []
