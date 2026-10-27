import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_classifier import metrics, parse_class_names, resolve_stage_epochs


class ClassifierTrainingTests(unittest.TestCase):
    def test_parse_class_names(self):
        self.assertEqual(
            parse_class_names("Clean, Stained"),
            ["Clean", "Stained"],
        )

    def test_parse_class_names_requires_two_classes(self):
        with self.assertRaises(ValueError):
            parse_class_names("Damaged")

    def test_metrics_include_both_class_recalls(self):
        result = metrics([0, 0, 1, 1], [0, 1, 1, 1], ["Damaged", "Not Damaged"])

        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["class_0_recall"], 0.5)
        self.assertAlmostEqual(result["class_1_recall"], 1.0)
        self.assertAlmostEqual(result["class_0_f1"], 2 / 3)

    def test_stage_epoch_resolution(self):
        self.assertEqual(resolve_stage_epochs(30, 0, 0), (0, 30))
        self.assertEqual(resolve_stage_epochs(30, 3, 12), (3, 12))


if __name__ == "__main__":
    unittest.main()
