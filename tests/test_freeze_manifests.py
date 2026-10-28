import csv
import json
from pathlib import Path

from src.freeze_manifests import freeze_manifests, sha256_file


def write_manifest(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_freeze_detects_cross_split_filename_and_source_overlap(tmp_path):
    columns = ["filename", "label", "source_group", "source_image", "sha256"]
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    write_manifest(
        train,
        [
            {
                "filename": "crop_a.jpg",
                "label": "0",
                "source_group": "s1",
                "source_image": "a.jpg",
                "sha256": "abc",
            }
        ],
        columns,
    )
    write_manifest(
        valid,
        [
            {
                "filename": "crop_b.jpg",
                "label": "0",
                "source_group": "s1",
                "source_image": "a.jpg",
                "sha256": "abc",
            }
        ],
        columns,
    )

    report = freeze_manifests(
        [train, valid], image_root=tmp_path, check_files=False
    )

    assert report["release_ready"] is False
    assert report["overlaps"]["source_groups"] == [
        {"value": "s1", "splits": ["train", "valid"]}
    ]
    assert report["overlaps"]["source_images"] == [
        {"value": "a.jpg", "splits": ["train", "valid"]}
    ]
    assert report["overlaps"]["sha256"] == [
        {"value": "abc", "splits": ["train", "valid"]}
    ]
    assert len(report["combined_sha256"]) == 64


def test_freeze_passes_when_no_overlap_and_files_exist(tmp_path):
    columns = ["filename", "label", "source_group", "source_image", "sha256"]
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    (tmp_path / "crop_a.jpg").write_bytes(b"a")
    (tmp_path / "crop_b.jpg").write_bytes(b"b")
    write_manifest(
        train,
        [
            {
                "filename": "crop_a.jpg",
                "label": "0",
                "source_group": "s1",
                "source_image": "a.jpg",
                "sha256": "abc",
            }
        ],
        columns,
    )
    write_manifest(
        valid,
        [
            {
                "filename": "crop_b.jpg",
                "label": "1",
                "source_group": "s2",
                "source_image": "b.jpg",
                "sha256": "def",
            }
        ],
        columns,
    )

    report = freeze_manifests([train, valid], image_root=tmp_path)

    assert report["release_ready"] is True
    assert report["overlap_count"] == 0
    assert report["missing_file_count"] == 0
    assert report["manifests"]["train"]["rows"] == 1
    assert report["manifests"]["train"]["sha256"] == sha256_file(train)


def test_freeze_records_missing_files(tmp_path):
    columns = ["filename", "label"]
    train = tmp_path / "train.csv"
    write_manifest(
        train,
        [{"filename": "missing.jpg", "label": "0"}],
        columns,
    )

    report = freeze_manifests([train], image_root=tmp_path)

    assert report["release_ready"] is False
    assert report["missing_file_count"] == 1
    assert report["manifests"]["train"]["missing_files"] == ["missing.jpg"]


def test_freeze_requires_unique_manifest_names(tmp_path):
    first = tmp_path / "one" / "train.csv"
    second = tmp_path / "two" / "train.csv"
    columns = ["filename", "label"]
    write_manifest(first, [{"filename": "a.jpg", "label": "0"}], columns)
    write_manifest(second, [{"filename": "b.jpg", "label": "0"}], columns)

    try:
        freeze_manifests([first, second], check_files=False)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Expected duplicate manifest name error")
