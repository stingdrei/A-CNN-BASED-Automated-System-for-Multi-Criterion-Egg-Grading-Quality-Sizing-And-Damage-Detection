import csv
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.csv_service import CSV_FIELDS, append_prediction, rows_for_prediction


def make_prediction(prediction_id: int = 1):
    boxes = [
        SimpleNamespace(
            egg_id=1,
            class_name="egg",
            confidence=0.91,
            x1=1,
            y1=2,
            x2=30,
            y2=40,
            size="Large",
            length_mm=62.5,
            breadth_mm=45.0,
            geometry_quality="bbox_fallback",
            weight_g=None,
            damage_status="Not Damaged",
            damage_confidence=0.88,
            cleanliness_status="Unavailable",
            cleanliness_confidence=None,
            grade="N/A",
            provisional_grade="A",
            requires_review=True,
            review_reason="cleanliness_unavailable;size_uncalibrated",
            detection_model_version="best.pt",
            damage_model_version="egg_grader.pth",
            cleanliness_model_version=None,
            calibration_version="uncalibrated",
        )
    ]
    return SimpleNamespace(
        id=prediction_id,
        tray_id="tray123",
        operator="alice",
        inference_metadata={
            "detection_threshold": 0.75,
            "damage_decision_threshold": 0.18,
            "cleanliness_decision_threshold": None,
            "tray_count_valid": True,
            "tray_over_capacity": False,
        },
        file_name="tray.jpg",
        captured_at=datetime(2026, 1, 2, 3, 4, 5),
        created_at=datetime(2026, 1, 2, 3, 4, 5),
        detection_boxes=boxes,
    )


class CsvServiceTests(unittest.TestCase):
    def test_rows_use_canonical_field_names(self):
        rows = rows_for_prediction(make_prediction())
        self.assertEqual(set(rows[0]), set(CSV_FIELDS))
        self.assertEqual(rows[0]["grade"], "N/A")
        self.assertEqual(rows[0]["provisional_grade"], "A")
        self.assertEqual(rows[0]["requires_review"], "true")
        self.assertEqual(rows[0]["cleanliness_status"], "Unavailable")
        self.assertEqual(rows[0]["operator"], "alice")
        self.assertEqual(rows[0]["damage_decision_threshold"], 0.18)
        self.assertEqual(rows[0]["tray_count_valid"], "true")

    def test_append_writes_header_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eggs.csv"
            append_prediction(make_prediction(1), str(path))
            append_prediction(make_prediction(2), str(path))
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            with path.open() as handle:
                header_count = sum(1 for line in handle if line.startswith("timestamp,"))
        self.assertEqual(len(rows), 2)
        self.assertEqual(header_count, 1)
        self.assertEqual(rows[0]["prediction_id"], "1")
        self.assertEqual(rows[1]["prediction_id"], "2")


if __name__ == "__main__":
    unittest.main()
