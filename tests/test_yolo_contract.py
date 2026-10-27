import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.ml.yolo_inference import YOLOPredictor


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, boxes=None, confs=None, classes=None):
        self.xyxy = FakeTensor(
            [[0, 50, 40, 90], [0, 10, 100, 60]]
            if boxes is None
            else boxes
        )
        self.conf = FakeTensor(
            [0.80, 0.95] if confs is None else confs
        )
        self.cls = FakeTensor(
            [0, 0] if classes is None else classes
        )

    def __len__(self):
        return len(self.conf.values)


class YoloContractTests(unittest.TestCase):
    def test_parse_results_emits_sorted_canonical_per_egg_fields(self):
        predictor = YOLOPredictor(
            model_path="dummy.pt", mm_per_pixel=0.09, weight_k=0.0
        )
        result = predictor._parse_results(SimpleNamespace(boxes=FakeBoxes()))
        self.assertEqual(result["total"], 2)
        self.assertEqual([det["egg_id"] for det in result["detections"]], [1, 2])
        first = result["detections"][0]
        self.assertEqual(first["bbox"], [0, 10, 100, 60])
        self.assertEqual(first["size"], "Small")
        self.assertNotIn("size_category", first)
        self.assertEqual(first["geometry_quality"], "bbox_fallback")
        self.assertEqual(first["grade"], "N/A")
        self.assertEqual(first["provisional_grade"], "N/A")
        self.assertEqual(first["detection_model_version"], "dummy.pt")
        self.assertEqual(first["calibration_version"], "uncalibrated")
        self.assertIsNone(first["weight_g"])
        self.assertAlmostEqual(first["length_mm"], 9.0)
        self.assertAlmostEqual(first["breadth_mm"], 4.5)
        self.assertTrue(result["count_valid"])
        self.assertEqual(result["count_status"], "ok")
        self.assertFalse(result["over_capacity"])
        self.assertEqual(result["max_eggs"], 5)

    def test_empty_tray_is_valid(self):
        predictor = YOLOPredictor(model_path="dummy.pt")
        result = predictor._parse_results(
            SimpleNamespace(boxes=FakeBoxes([], [], []))
        )
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["count_status"], "empty")
        self.assertTrue(result["count_valid"])
        self.assertFalse(result["over_capacity"])

    def test_five_boxes_are_valid_six_are_over_capacity(self):
        five = [
            [index * 20, 0, index * 20 + 10, 10]
            for index in range(5)
        ]
        predictor = YOLOPredictor(model_path="dummy.pt", max_eggs=5)
        result = predictor._parse_results(
            SimpleNamespace(
                boxes=FakeBoxes(five, [0.9] * 5, [0] * 5)
            )
        )
        self.assertEqual(result["total"], 5)
        self.assertTrue(result["count_valid"])
        self.assertFalse(result["over_capacity"])

        six = five + [[200, 0, 210, 10]]
        result = predictor._parse_results(
            SimpleNamespace(
                boxes=FakeBoxes(six, [0.9] * 6, [0] * 6)
            )
        )
        self.assertEqual(result["total"], 6)
        self.assertFalse(result["count_valid"])
        self.assertTrue(result["over_capacity"])
        self.assertEqual(result["count_status"], "over_capacity")

    def test_rejects_invalid_capacity_and_duplicate_threshold(self):
        with self.assertRaises(ValueError):
            YOLOPredictor(model_path="dummy.pt", max_eggs=0)
        with self.assertRaises(ValueError):
            YOLOPredictor(model_path="dummy.pt", duplicate_iou=1.5)

    def test_duplicate_boxes_are_removed_before_counting(self):
        predictor = YOLOPredictor(model_path="dummy.pt", duplicate_iou=0.85)
        result = predictor._parse_results(
            SimpleNamespace(
                boxes=FakeBoxes(
                    [[0, 0, 100, 100], [2, 2, 101, 101]],
                    [0.9, 0.8],
                    [0, 0],
                )
            )
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["duplicate_boxes_removed"], 1)


if __name__ == "__main__":
    unittest.main()
