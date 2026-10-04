import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.grading_service import calculate_grade


class GradingServiceTests(unittest.TestCase):
    def test_damaged_egg_is_rejected(self):
        self.assertEqual(
            calculate_grade("Damaged", "large", 0.95, 0.75),
            "Reject",
        )

    def test_low_confidence_damage_requires_review(self):
        self.assertEqual(
            calculate_grade("Damaged", "large", 0.70, 0.75),
            "Needs Review",
        )

    def test_unknown_damage_requires_review(self):
        self.assertEqual(
            calculate_grade(None, "large", 0.95, 0.75),
            "Needs Review",
        )

    def test_clean_size_maps_to_grade(self):
        self.assertEqual(
            calculate_grade("Not Damaged", "medium", 0.95, 0.75),
            "B",
        )


if __name__ == "__main__":
    unittest.main()
