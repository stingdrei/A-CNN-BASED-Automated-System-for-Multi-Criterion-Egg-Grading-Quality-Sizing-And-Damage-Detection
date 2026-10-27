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
    def __init__(self):
        self.xyxy = FakeTensor([[0, 50, 40, 90], [0, 10, 100, 60]])
        self.conf = FakeTensor([0.80, 0.95])
        self.cls = FakeTensor([0, 0])

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


if __name__ == "__main__":
    unittest.main()
