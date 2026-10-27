import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_damage_crops import padded_crop, safe_stem


class DamageCropTests(unittest.TestCase):
    def test_padded_crop_clamps_to_image_bounds(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        crop, bbox = padded_crop(image, [10, 20, 50, 60], 0.5)
        self.assertEqual(crop.shape[:2], (80, 70))
        self.assertEqual(bbox, (0, 0, 70, 80))

    def test_invalid_box_returns_none(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(padded_crop(image, [50, 50, 50, 60], 0.1))

    def test_safe_stem_removes_spaces_and_punctuation(self):
        self.assertEqual(
            safe_stem(Path("brokeneggs_2023-04-09 20_19_14.jpg")),
            "brokeneggs_2023-04-09_20_19_14",
        )


if __name__ == "__main__":
    unittest.main()
