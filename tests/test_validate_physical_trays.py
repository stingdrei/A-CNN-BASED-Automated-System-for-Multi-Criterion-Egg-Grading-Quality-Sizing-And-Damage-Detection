import csv
from pathlib import Path

import pytest

from src.validate_physical_trays import (
    evaluate_trays,
    read_expectations,
    write_csv,
)


class FakePredictor:
    def __init__(self, counts, errors=None):
        self.counts = counts
        self.errors = set(errors or [])

    def predict_image(self, path):
        name = Path(path).name
        if name in self.errors:
            raise RuntimeError("decode failed")
        count = self.counts[name]
        return {
            "total": count,
            "over_capacity": count > 5,
            "duplicate_boxes_removed": 0,
            "detections": [{}] * count,
        }


def expectation(name, count, image_path=None):
    return {
        "filename": name,
        "basename": Path(name).name,
        "expected_count": count,
        "notes": "",
        "image_path": image_path or name,
    }


def test_read_expectations_parses_and_rejects_duplicate_basenames(tmp_path):
    path = tmp_path / "trays.csv"
    path.write_text(
        "filename,expected_count,notes\n"
        "a/0_egg.jpg,0,empty\n"
        "b/5_egg.jpg,5,full\n"
    )
    rows = read_expectations(path)
    assert [row["expected_count"] for row in rows] == [0, 5]

    path.write_text(
        "filename,expected_count\n"
        "a/tray.jpg,0\n"
        "b/tray.jpg,5\n"
    )
    with pytest.raises(ValueError, match="Duplicate image basename"):
        read_expectations(path)


@pytest.mark.parametrize("value", ["-1", "five", ""])
def test_read_expectations_rejects_bad_counts(tmp_path, value):
    path = tmp_path / "trays.csv"
    path.write_text(f"filename,expected_count\ntray.jpg,{value}\n")
    with pytest.raises(ValueError):
        read_expectations(path)


def test_evaluate_trays_release_ready_when_zero_and_five_pass():
    rows = [
        expectation("0.jpg", 0),
        expectation("1.jpg", 1),
        expectation("5.jpg", 5),
    ]
    results, summary = evaluate_trays(
        rows,
        FakePredictor({"0.jpg": 0, "1.jpg": 1, "5.jpg": 5}),
    )

    assert summary["release_ready"] is True
    assert summary["exact_count_accuracy"] == 1.0
    assert summary["false_positives_per_tray"] == 0.0
    assert summary["missed_eggs_per_tray"] == 0.0
    assert summary["expected_empty_trays"] == 1
    assert summary["expected_five_egg_trays"] == 1
    assert all(row["exact"] for row in results)


def test_evaluate_trays_fails_when_zero_or_five_miscounts():
    rows = [
        expectation("0.jpg", 0),
        expectation("5.jpg", 5),
    ]
    _, summary = evaluate_trays(
        rows,
        FakePredictor({"0.jpg": 1, "5.jpg": 4}),
    )

    assert summary["release_ready"] is False
    assert summary["gates"]["zero_egg_trays_pass"]["pass"] is False
    assert summary["gates"]["five_egg_trays_pass"]["pass"] is False
    assert summary["exact_count_accuracy"] == 0.0


def test_evaluate_trays_requires_zero_and_five_coverage():
    rows = [
        expectation("1.jpg", 1),
        expectation("2.jpg", 2),
    ]
    _, summary = evaluate_trays(
        rows,
        FakePredictor({"1.jpg": 1, "2.jpg": 2}),
    )

    assert summary["release_ready"] is False
    assert summary["gates"]["zero_egg_trays_present"]["pass"] is False
    assert summary["gates"]["five_egg_trays_present"]["pass"] is False


def test_evaluate_trays_records_prediction_errors():
    rows = [expectation("0.jpg", 0), expectation("5.jpg", 5)]
    results, summary = evaluate_trays(
        rows,
        FakePredictor({"0.jpg": 0, "5.jpg": 5}, errors={"5.jpg"}),
    )

    assert summary["errors"] == 1
    assert summary["release_ready"] is False
    assert results[1]["error"] == "RuntimeError: decode failed"


def test_evaluate_trays_rejects_expectations_above_max():
    rows = [expectation("0.jpg", 0), expectation("5.jpg", 5), expectation("6.jpg", 6)]
    _, summary = evaluate_trays(
        rows,
        FakePredictor({"0.jpg": 0, "5.jpg": 5, "6.jpg": 6}),
    )

    assert summary["release_ready"] is False
    assert summary["gates"]["expectations_within_zero_to_max"]["pass"] is False


def test_write_csv(tmp_path):
    results, _ = evaluate_trays(
        [expectation("0.jpg", 0)],
        FakePredictor({"0.jpg": 0}),
    )
    path = tmp_path / "out.csv"
    write_csv(results, path)

    rows = list(csv.DictReader(path.open(newline="")))
    assert len(rows) == 1
    assert rows[0]["filename"] == "0.jpg"
    assert rows[0]["predicted_count"] == "0"
    assert rows[0]["exact"] == "True"
