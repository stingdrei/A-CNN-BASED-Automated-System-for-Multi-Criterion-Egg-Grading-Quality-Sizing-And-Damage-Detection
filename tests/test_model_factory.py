import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import EggGradingCNN
from model_factory import (
    ARCH_CUSTOM,
    ARCH_RESNET18,
    build_model,
    checkpoint_metadata,
)


class BuildModelTests(unittest.TestCase):
    def test_builds_custom_arch(self):
        model = build_model(ARCH_CUSTOM)

        self.assertIsInstance(model, EggGradingCNN)
        self.assertEqual(model.fc2.out_features, 2)

    def test_builds_resnet18_with_two_class_head(self):
        model = build_model(ARCH_RESNET18)

        self.assertEqual(model.fc.out_features, 2)

    def test_unknown_arch_raises(self):
        with self.assertRaises(ValueError):
            build_model("vgg16")

    def test_pretrained_download_failure_falls_back(self):
        import torchvision.models as tv_models

        real_resnet18 = tv_models.resnet18

        def offline_resnet18(weights=None, *args, **kwargs):
            if weights is not None:
                raise RuntimeError("pretrained weights unavailable offline")
            return real_resnet18(weights=None)

        with mock.patch(
            "torchvision.models.resnet18", side_effect=offline_resnet18
        ):
            with self.assertWarns(UserWarning):
                model = build_model(ARCH_RESNET18, pretrained=True)

        self.assertEqual(model.fc.out_features, 2)


class CheckpointMetadataTests(unittest.TestCase):
    def test_contains_arch_and_args(self):
        metadata = checkpoint_metadata(
            ARCH_RESNET18, num_classes=2, pretrained=False, seed=42
        )

        self.assertEqual(metadata["arch"], ARCH_RESNET18)
        self.assertEqual(
            metadata["arch_args"], {"num_classes": 2, "pretrained": False}
        )
        self.assertEqual(metadata["seed"], 42)

    def test_unknown_arch_raises(self):
        with self.assertRaises(ValueError):
            checkpoint_metadata("vgg16")


if __name__ == "__main__":
    unittest.main()
