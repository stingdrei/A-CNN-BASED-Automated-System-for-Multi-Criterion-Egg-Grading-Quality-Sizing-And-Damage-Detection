import sys
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.main import app
from app.schemas import (
    DetectionBoxResponse,
    GradeDistribution,
    HealthResponse,
    UserSettingsUpdate,
)


class ApiContractTests(unittest.TestCase):
    def test_detection_box_serializes_nested_confidence_geometry_versions(self):
        box = SimpleNamespace(
            id=1,
            prediction_id=2,
            egg_id=3,
            class_name="egg",
            confidence=0.91,
            x1=1,
            y1=2,
            x2=30,
            y2=40,
            annotated=True,
            grade="N/A",
            provisional_grade="A",
            size="Large",
            weight_g=None,
            damage_status="Not Damaged",
            damage_confidence=0.88,
            cleanliness_status="Unavailable",
            cleanliness_confidence=None,
            length_mm=62.5,
            breadth_mm=45.0,
            geometry_quality="bbox_fallback",
            detection_model_version="best.pt",
            damage_model_version="egg_grader.pth",
            cleanliness_model_version=None,
            calibration_version="uncalibrated",
            requires_review=True,
            review_reason="cleanliness_unavailable",
        )
        payload = DetectionBoxResponse.model_validate(box).model_dump()
        self.assertEqual(
            payload["confidence"],
            {
                "detection": 0.91,
                "damage": 0.88,
                "cleanliness": None,
            },
        )
        self.assertEqual(
            payload["geometry"],
            {
                "length_mm": 62.5,
                "breadth_mm": 45.0,
                "quality": "bbox_fallback",
            },
        )
        self.assertEqual(
            payload["model_versions"],
            {
                "detector": "best.pt",
                "damage": "egg_grader.pth",
                "cleanliness": None,
                "calibration": "uncalibrated",
            },
        )
        self.assertNotIn("detection_confidence", payload)
        self.assertEqual(payload["provisional_grade"], "A")
        self.assertTrue(payload["requires_review"])

    def test_grade_distribution_has_c_and_no_aa(self):
        payload = GradeDistribution(grade_a=1, grade_b=2, grade_c=3).model_dump()
        self.assertEqual(payload["grade_c"], 3)
        self.assertNotIn("grade_aa", payload)

    def test_documented_operation_aliases_are_registered(self):
        routes = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, "methods", set())
        }
        self.assertIn(("/api/v1/grade", "POST"), routes)
        self.assertIn(("/api/v1/records", "GET"), routes)
        self.assertIn(("/api/v1/stats", "GET"), routes)

    def test_upload_runs_in_threadpool_not_on_event_loop(self):
        from app.routers.predictions import upload_file

        self.assertFalse(inspect.iscoroutinefunction(upload_file))

    def test_settings_reject_non_positive_mm_per_pixel(self):
        from pydantic import ValidationError

        for invalid in (0, -0.09):
            with self.assertRaises(ValidationError):
                UserSettingsUpdate(mm_per_pixel=invalid)
        self.assertEqual(
            UserSettingsUpdate(mm_per_pixel=0.09).mm_per_pixel, 0.09
        )

    def test_legacy_aa_grade_normalizes_to_a(self):
        box = SimpleNamespace(
            id=1,
            prediction_id=2,
            egg_id=3,
            class_name="egg",
            confidence=0.91,
            x1=1,
            y1=2,
            x2=30,
            y2=40,
            annotated=False,
            grade="AA",
            provisional_grade="AA",
            size="Large",
            weight_g=None,
            damage_status="Not Damaged",
            damage_confidence=0.88,
            cleanliness_status="Unavailable",
            cleanliness_confidence=None,
            length_mm=62.5,
            breadth_mm=45.0,
            geometry_quality="bbox_fallback",
            detection_model_version="best.pt",
            damage_model_version="egg_grader.pth",
            cleanliness_model_version=None,
            calibration_version="uncalibrated",
            requires_review=False,
            review_reason=None,
        )
        payload = DetectionBoxResponse.model_validate(box).model_dump()
        self.assertEqual(payload["grade"], "A")
        self.assertEqual(payload["provisional_grade"], "A")

    def test_health_reports_cleanliness_as_unavailable(self):
        health = HealthResponse(
            status="degraded",
            model_loaded=True,
            damage_model_loaded=True,
            database_connected=False,
        )
        self.assertFalse(health.cleanliness_model_loaded)


if __name__ == "__main__":
    unittest.main()
