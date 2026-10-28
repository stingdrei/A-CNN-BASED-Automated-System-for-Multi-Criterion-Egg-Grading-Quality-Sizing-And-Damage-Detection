import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_cleanliness import (
    error_rows,
    expected_calibration_error,
    select_threshold,
    sweep_thresholds,
    threshold_metrics,
    validate_checkpoint_classes,
)


class CleanlinessEvaluationTests(unittest.TestCase):
    def test_validate_checkpoint_classes_accepts_clean_stained_order(self):
        self.assertEqual(
            validate_checkpoint_classes(["Clean", "Stained"]),
            ["Clean", "Stained"],
        )

    def test_validate_checkpoint_classes_rejects_wrong_order(self):
        with self.assertRaises(ValueError):
            validate_checkpoint_classes(["Stained", "Clean"])

    def test_threshold_metrics_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.9, 0.2])

        result = threshold_metrics(y_true, probabilities, 0.5)

        self.assertEqual(
            result["confusion_matrix"],
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["stained_recall"], 0.5)
        self.assertAlmostEqual(result["stained_precision"], 0.5)
        self.assertAlmostEqual(result["clean_recall"], 0.5)
        self.assertAlmostEqual(result["false_positive_rate"], 0.5)
        self.assertAlmostEqual(result["false_negative_rate"], 0.5)

    def test_select_threshold_respects_recall_floor(self):
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        sweep = sweep_thresholds(y_true, probabilities, step=0.1)

        selection = select_threshold(sweep, target_stained_recall=0.98)

        self.assertAlmostEqual(selection["selected_threshold"], 0.3)
        self.assertAlmostEqual(
            selection["selected_metrics"]["stained_recall"], 1.0
        )

    def test_select_threshold_falls_back_to_max_recall(self):
        y_true = np.array([0, 1])
        probabilities = np.array([0.9, 0.0])
        sweep = sweep_thresholds(y_true, probabilities, step=0.1)

        selection = select_threshold(sweep, target_stained_recall=1.0)

        self.assertIn("no threshold met the target", selection["note"])
        self.assertAlmostEqual(
            selection["selected_metrics"]["stained_recall"], 0.0
        )

    def test_expected_calibration_error_is_zero_for_perfect_confident_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        probabilities = np.array([0.0, 1.0, 0.0, 1.0])

        error = expected_calibration_error(y_true, probabilities, 0.5)

        self.assertAlmostEqual(error, 0.0)

    def test_error_rows_include_only_mistakes(self):
        labels = [("clean.jpg", 0), ("stain.jpg", 1), ("bad.jpg", 1)]
        probabilities = np.array([0.9, 0.8, 0.1])

        rows = error_rows("test", labels, probabilities, 0.5)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["outcome"], "false_positive")
        self.assertEqual(rows[1]["outcome"], "false_negative")
        self.assertAlmostEqual(
            float(rows[0]["stained_probability"]), 0.9
        )


if __name__ == "__main__":
    unittest.main()
