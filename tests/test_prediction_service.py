import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.database import Base
from app.models import User
from app.schemas import DetectionBoxResponse, PredictionDetail
from app.services.prediction_service import PredictionService


class FakePredictor:
    def __init__(self):
        self.confidence = 0.75

    def predict_image(self, image_path):
        return {
            "detections": [
                {
                    "egg_id": 1,
                    "class": "egg",
                    "confidence": 0.93,
                    "bbox": [1, 2, 30, 40],
                    "size": "Large",
                    "length_mm": 62.5,
                    "breadth_mm": 45.0,
                    "geometry_quality": "bbox_fallback",
                    "weight_g": None,
                    "grade": "N/A",
                    "provisional_grade": "N/A",
                    "detection_model_version": "best.pt",
                    "calibration_version": "uncalibrated",
                }
            ],
            "total": 1,
            "max_eggs": 5,
            "count_valid": True,
            "over_capacity": False,
            "count_status": "ok",
            "duplicate_boxes_removed": 0,
            "damaged": 0,
            "not_damaged": 0,
        }


class OverCapacityPredictor(FakePredictor):
    def predict_image(self, image_path):
        result = super().predict_image(image_path)
        result.update(
            max_eggs=5,
            count_valid=False,
            over_capacity=True,
            count_status="over_capacity",
        )
        return result


class EmptyPredictor:
    def __init__(self):
        self.confidence = 0.75

    def predict_image(self, image_path):
        return {
            "detections": [],
            "total": 0,
            "max_eggs": 5,
            "count_valid": True,
            "over_capacity": False,
            "count_status": "empty",
            "duplicate_boxes_removed": 0,
            "damaged": 0,
            "not_damaged": 0,
        }


class FakeDamage:
    model_version = "egg_grader.pth"

    def classify_crop(self, crop):
        return SimpleNamespace(
            status="Not Damaged",
            confidence=0.9,
            model_version="egg_grader.pth",
            review_reason=None,
        )


class FakeCleanliness:
    def classify_crop(self, crop):
        return SimpleNamespace(
            status="Stained",
            confidence=0.88,
            model_version="cleanliness_v1.pth",
            review_reason=None,
        )


class PredictionServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.user = User(
            email="test@example.com",
            username="tester",
            password_hash="x",
            is_active=True,
            mm_per_pixel=0.09,
        )
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)
        self.tmp = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tmp.name) / "tray.jpg"
        cv2.imwrite(str(self.image_path), np.zeros((80, 100, 3), dtype=np.uint8))

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        self.tmp.cleanup()

    def test_empty_tray_completes_with_zero_detections(self):
        service = PredictionService(EmptyPredictor(), FakeDamage())
        prediction = service.create_prediction(
            self.db,
            self.user.id,
            "empty-tray.jpg",
            str(self.image_path),
            "image",
        )

        with patch("app.services.prediction_service.append_prediction") as append:
            prediction = service.run_prediction(
                self.db, prediction.id, 0.75, True
            )
            append.assert_called_once()

        self.assertEqual(prediction.status, "completed")
        self.assertEqual(prediction.total_detections, 0)
        self.assertEqual(prediction.damaged_count, 0)
        self.assertEqual(prediction.not_damaged_count, 0)
        self.assertEqual(prediction.needs_review_count, 0)
        self.assertEqual(prediction.detection_boxes, [])
        self.assertIsNone(prediction.annotated_path)

    def test_stained_cleanliness_rejects_even_without_final_cleanliness(self):
        service = PredictionService(
            FakePredictor(), FakeDamage(), FakeCleanliness()
        )
        prediction = service.create_prediction(
            self.db,
            self.user.id,
            "stained.jpg",
            str(self.image_path),
            "image",
        )

        with patch("app.services.prediction_service.append_prediction"):
            prediction = service.run_prediction(
                self.db, prediction.id, 0.75, False
            )

        box = prediction.detection_boxes[0]
        self.assertEqual(box.cleanliness_status, "Stained")
        self.assertEqual(box.cleanliness_confidence, 0.88)
        self.assertEqual(box.cleanliness_model_version, "cleanliness_v1.pth")
        self.assertEqual(box.grade, "Reject")
        self.assertEqual(box.provisional_grade, "A")

    def test_prediction_persists_operator_and_inference_metadata(self):
        service = PredictionService(FakePredictor(), FakeDamage())
        prediction = service.create_prediction(
            self.db,
            self.user.id,
            "tray.jpg",
            str(self.image_path),
            "image",
            operator="alice",
        )

        with patch("app.services.prediction_service.append_prediction"):
            prediction = service.run_prediction(
                self.db, prediction.id, 0.75, False
            )

        self.assertEqual(prediction.operator, "alice")
        self.assertEqual(
            prediction.inference_metadata["detection_threshold"], 0.75
        )
        self.assertEqual(
            prediction.inference_metadata["detector_model_version"], None
        )
        self.assertEqual(
            prediction.inference_metadata["damage_model_version"],
            "egg_grader.pth",
        )
        self.assertTrue(prediction.inference_metadata["tray_count_valid"])
        self.assertFalse(prediction.inference_metadata["tray_over_capacity"])

    def test_over_capacity_marks_every_egg_for_review(self):
        service = PredictionService(OverCapacityPredictor(), FakeDamage())
        prediction = service.create_prediction(
            self.db,
            self.user.id,
            "six-eggs.jpg",
            str(self.image_path),
            "image",
        )

        with patch("app.services.prediction_service.append_prediction"):
            prediction = service.run_prediction(
                self.db, prediction.id, 0.75, False
            )

        self.assertFalse(prediction.inference_metadata["tray_count_valid"])
        self.assertTrue(prediction.inference_metadata["tray_over_capacity"])
        self.assertIn(
            "tray_over_capacity",
            prediction.detection_boxes[0].review_reason,
        )

    def test_completed_prediction_persists_provisional_contract(self):
        service = PredictionService(FakePredictor(), FakeDamage())
        prediction = service.create_prediction(
            self.db,
            self.user.id,
            "tray.jpg",
            str(self.image_path),
            "image",
        )
        self.assertIsNotNone(prediction.tray_id)
        self.assertIsNotNone(prediction.captured_at)

        with patch("app.services.prediction_service.append_prediction") as append:
            prediction = service.run_prediction(
                self.db, prediction.id, 0.75, False
            )
            append.assert_called_once()

        self.assertEqual(prediction.status, "completed")
        self.assertEqual(prediction.total_detections, 1)
        self.assertEqual(prediction.not_damaged_count, 1)
        self.assertEqual(prediction.needs_review_count, 1)
        self.assertEqual(len(prediction.detection_boxes), 1)

        box = prediction.detection_boxes[0]
        self.assertEqual(box.grade, "N/A")
        self.assertEqual(box.provisional_grade, "A")
        self.assertEqual(box.size, "Large")
        self.assertEqual(box.cleanliness_status, "Unavailable")
        self.assertTrue(box.requires_review)
        self.assertIn("cleanliness_unavailable", box.review_reason)
        self.assertIn("size_uncalibrated", box.review_reason)

        box_payload = DetectionBoxResponse.model_validate(box).model_dump()
        self.assertEqual(box_payload["confidence"]["detection"], 0.93)
        self.assertEqual(box_payload["confidence"]["damage"], 0.9)
        self.assertEqual(box_payload["provisional_grade"], "A")

        detail = PredictionDetail.model_validate(prediction).model_dump()
        self.assertEqual(detail["tray_id"], prediction.tray_id)
        self.assertEqual(len(detail["detection_boxes"]), 1)
        self.assertEqual(detail["detection_boxes"][0]["grade"], "N/A")


if __name__ == "__main__":
    unittest.main()
