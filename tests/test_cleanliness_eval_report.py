import csv
import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPORT_PATH = REPO_ROOT / "reports" / "cleanliness_crop_resnet18_v1_eval.json"
ERRORS_PATH = (
    REPO_ROOT / "reports" / "cleanliness_crop_resnet18_v1_eval_errors.csv"
)
CLASS_NAMES = ["Clean", "Stained"]
SPLITS = {"valid", "test"}
METRIC_KEYS = {
    "threshold",
    "samples",
    "confusion_matrix",
    "accuracy",
    "balanced_accuracy",
    "stained_precision",
    "stained_recall",
    "stained_f1",
    "macro_f1",
    "clean_recall",
    "false_positive_rate",
    "false_negative_rate",
    "expected_calibration_error",
}
ERROR_COLUMNS = [
    "split",
    "filename",
    "true_label",
    "predicted_label",
    "stained_probability",
    "outcome",
]


class CleanlinessEvalReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = json.loads(REPORT_PATH.read_text())
        with ERRORS_PATH.open(newline="") as handle:
            self.error_rows = list(csv.DictReader(handle))

    def test_report_records_portable_checkpoint_path(self):
        checkpoint = self.report["checkpoint"]

        self.assertNotIn("\\", checkpoint, "checkpoint path must be POSIX")
        self.assertNotRegex(checkpoint, r"^[A-Za-z]:", "no Windows drive prefix")
        self.assertEqual(checkpoint, Path(checkpoint).as_posix())
        self.assertTrue(checkpoint.endswith(".pth"))

    def test_report_records_device_and_checkpoint_identity(self):
        self.assertIsInstance(self.report["device"], str)
        self.assertTrue(self.report["device"].strip())
        self.assertRegex(self.report["checkpoint_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(self.report["class_names"], CLASS_NAMES)
        self.assertEqual(self.report["valid"]["samples"], 92)
        self.assertEqual(self.report["test"]["samples"], 91)

    def test_threshold_policy_is_internally_consistent(self):
        policy = self.report["threshold_policy"]
        sweep = self.report["threshold_sweep"]

        self.assertEqual(policy["selection_split"], "valid")
        self.assertGreater(len(sweep), 0)
        thresholds = [row["threshold"] for row in sweep]
        self.assertEqual(thresholds, sorted(set(thresholds)))
        self.assertIn(policy["selected_threshold"], thresholds)
        self.assertEqual(
            policy["selected_threshold"],
            policy["selected_metrics"]["threshold"],
        )
        self.assertGreaterEqual(policy["selected_metrics"]["stained_recall"], 0.98)

    def test_split_metrics_match_confusion_matrix(self):
        for split in ("valid", "test"):
            with self.subTest(split=split):
                metrics = self.report[split]
                self.assertTrue(METRIC_KEYS.issubset(metrics))
                matrix = metrics["confusion_matrix"]
                self.assertEqual(set(matrix), {"tn", "fp", "fn", "tp"})
                self.assertEqual(sum(matrix.values()), metrics["samples"])
                self.assertAlmostEqual(
                    metrics["accuracy"],
                    (matrix["tn"] + matrix["tp"]) / metrics["samples"],
                )
                self.assertAlmostEqual(
                    metrics["stained_recall"],
                    matrix["tp"] / (matrix["tp"] + matrix["fn"]),
                )
                for key in METRIC_KEYS - {"threshold", "samples", "confusion_matrix"}:
                    self.assertGreaterEqual(metrics[key], 0.0)
                    self.assertLessEqual(metrics[key], 1.0)

    def test_errors_csv_schema(self):
        self.assertGreater(len(self.error_rows), 0)
        self.assertEqual(list(self.error_rows[0]), ERROR_COLUMNS)
        for row in self.error_rows:
            with self.subTest(filename=row["filename"]):
                self.assertIn(row["split"], SPLITS)
                self.assertIn(row["true_label"], CLASS_NAMES)
                self.assertIn(row["predicted_label"], CLASS_NAMES)
                self.assertIn(
                    row["outcome"], {"false_positive", "false_negative"}
                )
                self.assertNotEqual(row["true_label"], row["predicted_label"])
                self.assertNotIn("\\", row["filename"])
                self.assertFalse(Path(row["filename"]).is_absolute())
                probability = float(row["stained_probability"])
                self.assertGreaterEqual(probability, 0.0)
                self.assertLessEqual(probability, 1.0)
                expected_outcome = (
                    "false_positive"
                    if row["true_label"] == "Clean"
                    else "false_negative"
                )
                self.assertEqual(row["outcome"], expected_outcome)
                predicted_stained = probability >= self.report["threshold_policy"][
                    "selected_threshold"
                ]
                self.assertEqual(
                    row["predicted_label"], "Stained" if predicted_stained else "Clean"
                )

    def test_errors_csv_matches_report_confusion_matrix(self):
        for split in sorted(SPLITS):
            with self.subTest(split=split):
                matrix = self.report[split]["confusion_matrix"]
                outcomes = [
                    row["outcome"] for row in self.error_rows if row["split"] == split
                ]
                self.assertEqual(outcomes.count("false_positive"), matrix["fp"])
                self.assertEqual(outcomes.count("false_negative"), matrix["fn"])

    def test_no_environment_specific_paths_in_artifacts(self):
        for path in (REPORT_PATH, ERRORS_PATH):
            text = path.read_text()
            self.assertIsNone(
                re.search(r"[A-Za-z]:\\\\|\\\\Users\\\\|/Users/", text),
                f"{path.name} leaks an absolute local path",
            )


if __name__ == "__main__":
    unittest.main()
