import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibrate_weight import fit_calibration, load_records


class WeightCalibrationTests(unittest.TestCase):
    def test_recovers_known_coefficient(self):
        length = np.array([55.0, 58.0, 60.0, 62.0, 65.0, 57.0, 59.0, 61.0, 63.0, 64.0])
        breadth = np.array([42.0, 43.0, 44.0, 45.0, 46.0, 42.5, 43.5, 44.5, 45.5, 46.0])
        k = 0.00052
        actual = k * length * np.square(breadth)

        result = fit_calibration(length, breadth, actual)

        self.assertAlmostEqual(result["k"], k, places=12)
        self.assertAlmostEqual(result["mae_g"], 0.0, places=10)
        self.assertAlmostEqual(result["r2"], 1.0, places=10)
        self.assertAlmostEqual(result["within_3g_rate"], 1.0)
        self.assertEqual(result["samples"], 10)

    def test_rejects_zero_axes(self):
        with self.assertRaises(ValueError):
            fit_calibration([0.0], [44.0], [60.0])

    def test_loads_weight_alias_and_rejects_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.csv"
            path.write_text(
                "length_mm,breadth_mm,weight_g\n"
                "60,45,60.0\n"
                "61,46,62.0\n"
            )
            length, breadth, actual = load_records(path)

            self.assertEqual(length.tolist(), [60.0, 61.0])
            self.assertEqual(breadth.tolist(), [45.0, 46.0])
            self.assertEqual(actual.tolist(), [60.0, 62.0])

            path.write_text("length_mm,breadth_mm,weight_g\n60,0,60.0\n")
            with self.assertRaises(ValueError):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
