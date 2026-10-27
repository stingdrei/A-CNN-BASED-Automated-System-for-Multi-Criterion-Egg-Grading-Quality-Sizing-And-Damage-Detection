import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tray_count import box_iou, deduplicate_boxes, tray_count_status


class TrayCountTests(unittest.TestCase):
    def test_empty_tray_is_valid_but_not_over_capacity(self):
        status = tray_count_status(0, 5)
        self.assertTrue(status["count_valid"])
        self.assertFalse(status["over_capacity"])
        self.assertEqual(status["count_status"], "empty")

    def test_five_is_valid_six_is_over_capacity(self):
        self.assertTrue(tray_count_status(5, 5)["count_valid"])
        self.assertEqual(tray_count_status(5, 5)["count_status"], "ok")
        six = tray_count_status(6, 5)
        self.assertFalse(six["count_valid"])
        self.assertTrue(six["over_capacity"])
        self.assertEqual(six["count_status"], "over_capacity")

    def test_duplicate_boxes_keep_highest_confidence_and_reading_order(self):
        candidates = [
            {"bbox": [100, 0, 200, 100], "confidence": 0.90},
            {"bbox": [0, 0, 100, 100], "confidence": 0.80},
            {"bbox": [101, 1, 200, 101], "confidence": 0.95},
        ]
        kept, removed = deduplicate_boxes(candidates, 0.85)
        self.assertEqual(removed, 1)
        self.assertEqual([item["confidence"] for item in kept], [0.80, 0.95])

    def test_box_iou_matches_known_overlap(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 5, 10]), 0.5)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertEqual(box_iou([0, 0, 0, 10], [0, 0, 1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
