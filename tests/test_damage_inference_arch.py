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

from app.ml.damage_inference import DamageClassifier, decide_damage
from src.model_factory import build_model, checkpoint_metadata


class DamageInferenceArchTests(unittest.TestCase):
    def _write_checkpoint(
        self,
        path: Path,
        arch: str,
        include_arch: bool = True,
        damaged_threshold: Optional[float] = None,
    ) -> None:
        model = build_model(arch, num_classes=2, pretrained=False)
        checkpoint = {
            "state_dict": model.state_dict(),
            "class_names": ["Damaged", "Not Damaged"],
            "img_size": 224,
        }
        if damaged_threshold is not None:
            checkpoint["damaged_threshold"] = damaged_threshold
        if include_arch:
            checkpoint.update(
                checkpoint_metadata(arch, num_classes=2, pretrained=False)
            )
        torch.save(checkpoint, path)

    def test_loads_custom_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.pth"
            self._write_checkpoint(path, "custom")
            classifier = DamageClassifier(
                checkpoint_path=str(path),
                review_threshold=0.0,
                device="cpu",
            )

            self.assertTrue(classifier.load_model())
            self.assertEqual(classifier.arch, "custom")
            crop = np.zeros((32, 32, 3), dtype=np.uint8)
            self.assertIn(
                classifier.classify_crop(crop).status,
                classifier.EXPECTED_CLASSES,
            )

    def test_loads_resnet18_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resnet18.pth"
            self._write_checkpoint(path, "resnet18")
            classifier = DamageClassifier(
                checkpoint_path=str(path),
                review_threshold=0.0,
                device="cpu",
            )

            self.assertTrue(classifier.load_model())
            self.assertEqual(classifier.arch, "resnet18")

    def test_missing_arch_falls_back_to_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pth"
            self._write_checkpoint(path, "custom", include_arch=False)
            classifier = DamageClassifier(
                checkpoint_path=str(path),
                review_threshold=0.0,
                device="cpu",
            )

            self.assertTrue(classifier.load_model())
            self.assertEqual(classifier.arch, "custom")

    def test_unknown_arch_fails_to_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pth"
            model = build_model("custom", num_classes=2)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": ["Damaged", "Not Damaged"],
                    "img_size": 224,
                    "arch": "vgg16",
                },
                path,
            )
            classifier = DamageClassifier(
                checkpoint_path=str(path),
                device="cpu",
            )

            self.assertFalse(classifier.load_model())

    def test_checkpoint_threshold_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.pth"
            self._write_checkpoint(path, "custom", damaged_threshold=0.18)
            classifier = DamageClassifier(
                checkpoint_path=str(path),
                review_threshold=0.0,
                device="cpu",
            )

            self.assertTrue(classifier.load_model())
            self.assertEqual(classifier.damaged_threshold, 0.18)

    def test_explicit_threshold_overrides_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.pth"
            self._write_checkpoint(path, "custom", damaged_threshold=0.18)
            classifier = DamageClassifier(
                checkpoint_path=str(path),
                review_threshold=0.0,
                damaged_threshold=0.4,
                device="cpu",
            )

            self.assertTrue(classifier.load_model())
            self.assertEqual(classifier.damaged_threshold, 0.4)

    def test_damaged_is_not_downgraded_to_review(self):
        status, confidence, reason = decide_damage(0.2, 0.18, 0.75)

        self.assertEqual(status, "Damaged")
        self.assertAlmostEqual(confidence, 0.2)
        self.assertIsNone(reason)

    def test_uncertain_not_damaged_routes_to_review(self):
        status, confidence, reason = decide_damage(0.3, 0.5, 0.75)

        self.assertEqual(status, "Needs Review")
        self.assertAlmostEqual(confidence, 0.7)
        self.assertEqual(reason, "low_confidence")

    def test_confident_not_damaged_is_graded(self):
        status, confidence, reason = decide_damage(0.1, 0.18, 0.75)

        self.assertEqual(status, "Not Damaged")
        self.assertAlmostEqual(confidence, 0.9)
        self.assertIsNone(reason)

    def test_low_confidence_routes_to_needs_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.pth"
            self._write_checkpoint(path, "custom")
            classifier = DamageClassifier(
                checkpoint_path=str(path),
                review_threshold=1.1,
                damaged_threshold=1.0,
                device="cpu",
            )

            result = classifier.classify_crop(
                np.zeros((32, 32, 3), dtype=np.uint8)
            )

            self.assertEqual(result.status, "Needs Review")
            self.assertEqual(result.review_reason, "low_confidence")

    def test_empty_crop_does_not_load_model(self):
        classifier = DamageClassifier(
            checkpoint_path="/missing/checkpoint.pth",
            device="cpu",
        )

        result = classifier.classify_crop(np.empty((0, 0, 3), dtype=np.uint8))

        self.assertEqual(result.status, "Needs Review")
        self.assertEqual(result.review_reason, "empty_crop")


if __name__ == "__main__":
    unittest.main()
