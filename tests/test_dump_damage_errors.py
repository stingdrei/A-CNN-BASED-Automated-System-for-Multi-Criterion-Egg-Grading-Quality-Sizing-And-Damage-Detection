import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dump_damage_errors import classify_probability, error_type, resolve_checkpoint_config


class DumpDamageErrorsTests(unittest.TestCase):
    def test_resolves_resnet_checkpoint_metadata(self):
        config = resolve_checkpoint_config(
            {
                "arch": "resnet18",
                "arch_args": {"num_classes": 2, "pretrained": False},
                "class_names": ["Damaged", "Not Damaged"],
                "img_size": 224,
                "damaged_threshold": 0.18,
            }
        )
        self.assertEqual(config["arch"], "resnet18")
        self.assertEqual(config["num_classes"], 2)
        self.assertEqual(config["damaged_threshold"], 0.18)
        self.assertEqual(config["damaged_threshold_source"], "checkpoint")

    def test_cli_threshold_and_image_size_override(self):
        config = resolve_checkpoint_config(
            {
                "arch": "custom",
                "class_names": ["Damaged", "Not Damaged"],
                "img_size": 224,
                "damaged_threshold": 0.18,
            },
            img_size=256,
            threshold=0.4,
        )
        self.assertEqual(config["img_size"], 256)
        self.assertEqual(config["damaged_threshold"], 0.4)
        self.assertEqual(config["damaged_threshold_source"], "cli")

    def test_rejects_unknown_class_order(self):
        with self.assertRaises(ValueError):
            resolve_checkpoint_config(
                {"class_names": ["Clean", "Stained"], "arch": "custom"}
            )

    def test_classifies_at_and_above_threshold_as_damaged(self):
        self.assertEqual(classify_probability(0.18, 0.18), (0, 0.18))
        self.assertEqual(classify_probability(0.179, 0.18), (1, 0.821))
        self.assertEqual(
            error_type(true_label=1, pred_label=0),
            "FP",
        )


if __name__ == "__main__":
    unittest.main()
