import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dump_damage_errors import CSV_COLUMNS, ERROR_TYPES, collect_errors, error_type
from model import EggGradingCNN


class ErrorTypeTests(unittest.TestCase):
    def test_maps_all_four_cases(self):
        self.assertEqual(error_type(0, 0), "TP")
        self.assertEqual(error_type(1, 1), "TN")
        self.assertEqual(error_type(0, 1), "FN")
        self.assertEqual(error_type(1, 0), "FP")

    def test_error_type_values_are_known(self):
        for true_label in (0, 1):
            for pred_label in (0, 1):
                self.assertIn(error_type(true_label, pred_label), ERROR_TYPES)


class CollectErrorsTests(unittest.TestCase):
    def _build_fixture(self, root: Path):
        image_names = ["egg_a.jpg", "egg_b.jpg"]
        for name in image_names:
            image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(root / name), image)

        manifest = root / "manifest.csv"
        pd.DataFrame(
            {"filename": image_names, "label": [0, 1]}
        ).to_csv(manifest, index=False)

        checkpoint = root / "checkpoint.pth"
        model = EggGradingCNN(num_classes=2)
        torch.save({"state_dict": model.state_dict()}, checkpoint)
        return manifest, checkpoint, image_names

    def test_emits_expected_columns_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, checkpoint, image_names = self._build_fixture(root)

            frame = pd.DataFrame(
                collect_errors(
                    [(name, label) for name, label in zip(image_names, [0, 1])],
                    str(root),
                    str(checkpoint),
                    batch_size=2,
                    device=torch.device("cpu"),
                )
            )

            self.assertEqual(list(frame.columns), CSV_COLUMNS)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame["filename"].tolist(), image_names)
            self.assertTrue(frame["error_type"].isin(ERROR_TYPES).all())
            self.assertTrue(frame["confidence"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
