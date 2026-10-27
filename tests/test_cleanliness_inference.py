import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from app.ml.cleanliness_inference import (
    CleanlinessClassifier,
    decide_cleanliness,
)
from src.model_factory import build_model, checkpoint_metadata


class CleanlinessInferenceTests(unittest.TestCase):
    def _write_checkpoint(
        self,
        path: Path,
        arch: str = "custom",
        stained_threshold: Optional[float] = None,
    ) -> None:
        model = build_model(arch, num_classes=2, pretrained=False)
        checkpoint = {
            "state_dict": model.state_dict(),
            "class_names": ["Clean", "Stained"],
            "img_size": 224,
        }
        checkpoint.update(
            checkpoint_metadata(arch, num_classes=2, pretrained=False)
        )
        if stained_threshold is not None:
            checkpoint["stained_threshold"] = stained_threshold
        torch.save(checkpoint, path)

    def test_loads_checkpoint_and_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cleanliness.pth"
            self._write_checkpoint(path, stained_threshold=0.42)
            classifier = CleanlinessClassifier(
                checkpoint_path=str(path),
                review_threshold=0.0,
                device="cpu",
            )

            self.assertTrue(classifier.load_model())
            self.assertEqual(classifier.arch, "custom")
            self.assertEqual(classifier.stained_threshold, 0.42)

    def test_explicit_threshold_overrides_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cleanliness.pth"
            self._write_checkpoint(path, stained_threshold=0.42)
            classifier = CleanlinessClassifier(
                checkpoint_path=str(path),
                review_threshold=0.0,
                stained_threshold=0.3,
                device="cpu",
            )

            self.assertTrue(classifier.load_model())
            self.assertEqual(classifier.stained_threshold, 0.3)

    def test_stained_is_not_downgraded_to_review(self):
        status, confidence, reason = decide_cleanliness(0.25, 0.2, 0.75)

        self.assertEqual(status, "Stained")
        self.assertAlmostEqual(confidence, 0.25)
        self.assertIsNone(reason)

    def test_uncertain_clean_routes_to_review(self):
        status, confidence, reason = decide_cleanliness(0.3, 0.5, 0.75)

        self.assertEqual(status, "Needs Review")
        self.assertAlmostEqual(confidence, 0.7)
        self.assertEqual(reason, "low_confidence")

    def test_confident_clean_is_graded(self):
        status, confidence, reason = decide_cleanliness(0.1, 0.2, 0.75)

        self.assertEqual(status, "Clean")
        self.assertAlmostEqual(confidence, 0.9)
        self.assertIsNone(reason)

    def test_missing_checkpoint_is_not_loaded(self):
        classifier = CleanlinessClassifier(
            checkpoint_path="/missing/cleanliness.pth",
            device="cpu",
        )

        self.assertFalse(classifier.load_model())

    def test_empty_crop_does_not_load_model(self):
        classifier = CleanlinessClassifier(
            checkpoint_path="/missing/cleanliness.pth",
            device="cpu",
        )

        result = classifier.classify_crop(np.empty((0, 0, 3), dtype=np.uint8))

        self.assertEqual(result.status, "Needs Review")
        self.assertEqual(result.review_reason, "empty_crop")


if __name__ == "__main__":
    unittest.main()
