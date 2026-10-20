import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model_factory import ARCH_CUSTOM, ARCH_RESNET18, build_model
from train_classifier import (
    build_checkpoint,
    resolve_stage_epochs,
    set_backbone_frozen,
    set_frozen_bn_eval,
    unfreeze_all,
)


class StageEpochTests(unittest.TestCase):
    def test_legacy_fallback_uses_single_full_stage(self):
        self.assertEqual(resolve_stage_epochs(30, 0, 0), (0, 30))

    def test_explicit_stages_win(self):
        self.assertEqual(resolve_stage_epochs(30, 2, 3), (2, 3))

    def test_negative_stage_raises(self):
        with self.assertRaises(ValueError):
            resolve_stage_epochs(30, -1, 2)


class FreezeTests(unittest.TestCase):
    def test_resnet18_stage1_freezes_backbone(self):
        model = build_model(ARCH_RESNET18)

        set_backbone_frozen(model, ARCH_RESNET18)

        self.assertTrue(model.fc.weight.requires_grad)
        self.assertFalse(model.conv1.weight.requires_grad)
        self.assertFalse(model.layer1[0].conv1.weight.requires_grad)

    def test_resnet18_unfreeze_all(self):
        model = build_model(ARCH_RESNET18)
        set_backbone_frozen(model, ARCH_RESNET18)

        unfreeze_all(model)

        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_custom_stage1_freezes_backbone(self):
        model = build_model(ARCH_CUSTOM)

        set_backbone_frozen(model, ARCH_CUSTOM)

        self.assertTrue(model.fc2.weight.requires_grad)
        self.assertFalse(model.conv1.weight.requires_grad)
        self.assertFalse(model.fc1.weight.requires_grad)

    def test_frozen_bn_is_eval(self):
        model = build_model(ARCH_RESNET18)
        model.train()

        set_frozen_bn_eval(model)

        self.assertFalse(model.bn1.training)


class CheckpointTests(unittest.TestCase):
    def test_metadata_and_reload(self):
        model = build_model(ARCH_RESNET18)
        checkpoint = build_checkpoint(
            model,
            ARCH_RESNET18,
            pretrained=False,
            img_size=224,
            seed=42,
            record={"epoch": 1, "val_loss": 0.5, "f1": 0.6},
            hyperparams={"lr": 1e-4, "stage1_epochs": 1, "stage2_epochs": 1},
        )

        self.assertEqual(checkpoint["arch"], ARCH_RESNET18)
        self.assertEqual(
            checkpoint["arch_args"], {"num_classes": 2, "pretrained": False}
        )
        self.assertEqual(checkpoint["class_names"], ["Damaged", "Not Damaged"])
        self.assertEqual(checkpoint["hyperparams"]["lr"], 1e-4)

        restored = build_model(checkpoint["arch"], num_classes=2)
        restored.load_state_dict(checkpoint["state_dict"])

    def test_custom_checkpoint_keeps_legacy_keys(self):
        model = build_model(ARCH_CUSTOM)
        checkpoint = build_checkpoint(
            model, ARCH_CUSTOM, False, 224, 42, {}, {}
        )

        for key in (
            "state_dict",
            "class_names",
            "img_size",
            "seed",
            "best_val_metrics",
        ):
            self.assertIn(key, checkpoint)
        self.assertEqual(checkpoint["arch"], ARCH_CUSTOM)


if __name__ == "__main__":
    unittest.main()
