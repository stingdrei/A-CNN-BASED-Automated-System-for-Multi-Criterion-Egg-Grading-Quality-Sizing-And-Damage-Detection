import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.grading_service import (
    calculate_grade,
    collect_review_reasons,
    join_review_reasons,
    map_to_grade,
    provisional_grade,
)


class GradingServiceTests(unittest.TestCase):
    def test_damaged_egg_is_rejected(self):
        self.assertEqual(
            calculate_grade("Damaged", "Large", 0.95, 0.75),
            "Reject",
        )

    def test_damaged_egg_has_reject_provisional_grade(self):
        self.assertEqual(provisional_grade("Damaged", "Large"), "Reject")

    def test_low_detection_confidence_is_not_graded(self):
        self.assertEqual(
            calculate_grade("Not Damaged", "Large", 0.70, 0.75),
            "N/A",
        )

    def test_damaged_low_confidence_still_rejects(self):
        self.assertEqual(
            calculate_grade(
                "Damaged",
                "Large",
                0.95,
                0.75,
                damage_confidence=0.20,
                damage_threshold=0.75,
            ),
            "Reject",
        )

    def test_not_damaged_low_confidence_is_not_graded(self):
        self.assertEqual(
            calculate_grade(
                "Not Damaged",
                "Large",
                0.95,
                0.75,
                damage_confidence=0.60,
                damage_threshold=0.75,
            ),
            "N/A",
        )

    def test_unknown_damage_is_not_graded(self):
        self.assertEqual(
            calculate_grade(None, "Large", 0.95, 0.75),
            "N/A",
        )

    def test_clean_size_maps_to_grade(self):
        self.assertEqual(
            calculate_grade(
                "Not Damaged",
                "Medium",
                0.95,
                0.75,
                cleanliness_status="Clean",
            ),
            "B",
        )

    def test_default_cleanliness_is_not_assumed_clean(self):
        self.assertEqual(
            calculate_grade("Not Damaged", "Large", 0.95, 0.75),
            "N/A",
        )

    def test_unavailable_cleanliness_blocks_final_grade(self):
        self.assertEqual(
            map_to_grade("Not Damaged", "Unavailable", "Large"),
            "N/A",
        )
        self.assertEqual(
            provisional_grade("Not Damaged", "Large"),
            "A",
        )

    def test_stained_egg_is_rejected(self):
        self.assertEqual(map_to_grade("Not Damaged", "Stained", "Large"), "Reject")

    def test_review_reasons_include_unavailable_criteria(self):
        reasons = collect_review_reasons(
            detection_confidence=0.95,
            detection_threshold=0.75,
            damage_status="Not Damaged",
            damage_confidence=0.90,
            damage_threshold=0.75,
            cleanliness_status="Unavailable",
            size="Large",
            weight_g=None,
        )
        self.assertEqual(
            reasons,
            [
                "cleanliness_unavailable",
            ],
        )
        self.assertEqual(
            join_review_reasons(reasons),
            "cleanliness_unavailable",
        )

    def test_uncalibrated_weight_is_not_a_per_egg_review_reason(self):
        reasons = collect_review_reasons(
            detection_confidence=0.95,
            detection_threshold=0.75,
            damage_status="Not Damaged",
            damage_confidence=0.90,
            damage_threshold=0.75,
            cleanliness_status="Clean",
            cleanliness_confidence=0.95,
            size="Large",
            weight_g=None,
            weight_calibrated=False,
        )
        self.assertEqual(reasons, [])

    def test_missing_weight_after_calibration_requires_review(self):
        reasons = collect_review_reasons(
            detection_confidence=0.95,
            detection_threshold=0.75,
            damage_status="Not Damaged",
            damage_confidence=0.90,
            damage_threshold=0.75,
            cleanliness_status="Clean",
            cleanliness_confidence=0.95,
            size="Large",
            weight_g=None,
            weight_calibrated=True,
        )
        self.assertEqual(reasons, ["weight_unavailable"])

    def test_low_damage_confidence_requires_review(self):
        reasons = collect_review_reasons(
            detection_confidence=0.95,
            detection_threshold=0.75,
            damage_status="Damaged",
            damage_confidence=0.60,
            damage_threshold=0.75,
            cleanliness_status="Clean",
            cleanliness_confidence=0.95,
            size="Large",
            weight_g=60.0,
        )
        self.assertEqual(reasons, ["low_damage_confidence"])


if __name__ == "__main__":
    unittest.main()
