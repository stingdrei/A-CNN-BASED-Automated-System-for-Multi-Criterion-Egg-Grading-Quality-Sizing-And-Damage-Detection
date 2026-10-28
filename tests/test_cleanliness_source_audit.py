import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit_cleanliness_sources import (
    audit_source,
    box_to_xyxy,
    label_for_category,
    numeric_summary,
    stock_keyword,
)


class CleanlinessSourceAuditTests(unittest.TestCase):
    def test_label_mapping_handles_double_spaces_and_weak_white(self):
        self.assertEqual(label_for_category("Brownegg"), (0, "Clean", False))
        self.assertEqual(
            label_for_category("dirt stained white  egg"),
            (1, "Stained", False),
        )
        self.assertEqual(label_for_category("white egg"), (0, "Clean", True))
        self.assertIsNone(label_for_category("color"))
        self.assertIsNone(label_for_category("eggs"))

    def test_box_to_xyxy_clips_and_rejects_invalid(self):
        self.assertEqual(box_to_xyxy([-5, -5, 20, 20], 100, 100), (0, 0, 15, 15))
        self.assertEqual(box_to_xyxy([90, 90, 20, 20], 100, 100), (90, 90, 100, 100))
        self.assertIsNone(box_to_xyxy([0, 0, 0, 10], 100, 100))
        self.assertIsNone(box_to_xyxy([120, 120, 10, 10], 100, 100))
        self.assertIsNone(box_to_xyxy([0, 0, 10], 100, 100))

    def test_numeric_summary(self):
        self.assertEqual(numeric_summary([]), {"count": 0})
        summary = numeric_summary([1, 2, 3, 4])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["mean"], 2.5)

    def test_stock_keyword(self):
        self.assertEqual(stock_keyword("depositphotos_123.jpg"), "depositphotos")
        self.assertEqual(stock_keyword("clean_egg.jpg"), None)

    def test_audit_source_counts_mapped_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            split_dir = root / "train"
            split_dir.mkdir(parents=True)
            Image.new("RGB", (100, 100), "white").save(split_dir / "egg.jpg")
            payload = {
                "categories": [
                    {"id": 0, "name": "Brownegg"},
                    {"id": 1, "name": "dirt stained white  egg"},
                ],
                "images": [
                    {
                        "id": 1,
                        "file_name": "egg.jpg",
                        "width": 100,
                        "height": 100,
                    }
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20]},
                    {"id": 11, "image_id": 1, "category_id": 1, "bbox": [50, 50, 20, 20]},
                ],
            }
            (split_dir / "_annotations.coco.json").write_text(json.dumps(payload))

            report, images, crops = audit_source(root, min_crop_size=8)

            self.assertEqual(report["images"], 1)
            self.assertEqual(report["annotations"], 2)
            self.assertEqual(report["mapped_counts"], {"Clean": 1, "Stained": 1})
            self.assertEqual(len(images), 1)
            self.assertEqual(len(crops), 2)
            self.assertEqual({row["label_name"] for row in crops}, {"Clean", "Stained"})


if __name__ == "__main__":
    unittest.main()
