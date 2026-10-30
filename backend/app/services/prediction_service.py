import os
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from uuid import uuid4
from sqlalchemy.orm import Session
import cv2

from app.models import Prediction, DetectionBox
from app.ml.yolo_inference import YOLOPredictor
from app.config import settings
from app.ml.cleanliness_inference import CleanlinessClassifier
from app.ml.damage_inference import DamageClassifier
from app.services.csv_service import append_prediction
from app.services.grading_service import (
    calculate_grade,
    collect_review_reasons,
    join_review_reasons,
    provisional_grade,
)


class PredictionService:
    def __init__(
        self,
        predictor: Optional[YOLOPredictor] = None,
        damage_classifier: Optional[DamageClassifier] = None,
        cleanliness_classifier: Optional[CleanlinessClassifier] = None,
    ):
        self.predictor = predictor or YOLOPredictor()
        self.damage_classifier = damage_classifier
        self.cleanliness_classifier = cleanliness_classifier

    def create_prediction(
        self,
        db: Session,
        user_id: int,
        file_name: str,
        file_path: str,
        file_type: str,
        operator: Optional[str] = None,
    ) -> Prediction:
        prediction = Prediction(
            user_id=user_id,
            file_name=file_name,
            file_type=file_type,
            file_path=file_path,
            tray_id=uuid4().hex,
            captured_at=datetime.utcnow(),
            operator=operator,
            status="processing",
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction

    def run_prediction(
        self,
        db: Session,
        prediction_id: int,
        confidence: float,
        save_annotated: bool,
        mm_per_pixel: Optional[float] = None,
    ) -> Prediction:
        start_time = time.time()

        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise ValueError("Prediction not found")

        img = None

        try:
            if prediction.file_type != "image":
                raise ValueError("Video prediction is not supported yet")

            img = cv2.imread(prediction.file_path)
            if img is None:
                raise ValueError("Uploaded image could not be decoded")
            result = self.predictor.predict_image(
                prediction.file_path,
                confidence=confidence,
                mm_per_pixel=mm_per_pixel,
            )
            if self.damage_classifier is None:
                raise RuntimeError("Damage classifier is not configured")
            damage_review_threshold = getattr(
                self.damage_classifier,
                "review_threshold",
                settings.DAMAGE_REVIEW_THRESHOLD,
            )

            for det in result["detections"]:
                crop = crop_detection(
                    img,
                    det["bbox"],
                    settings.DETECTION_CROP_PADDING,
                    settings.MIN_CROP_SIZE,
                )
                damage = self.damage_classifier.classify_crop(crop)
                det["damage_status"] = damage.status
                det["damage_confidence"] = damage.confidence
                det["damage_model_version"] = damage.model_version
                if self.cleanliness_classifier is None:
                    cleanliness_status = "Unavailable"
                    cleanliness_confidence = None
                    cleanliness_model_version = None
                else:
                    cleanliness = self.cleanliness_classifier.classify_crop(crop)
                    cleanliness_status = cleanliness.status
                    cleanliness_confidence = cleanliness.confidence
                    cleanliness_model_version = cleanliness.model_version

                det["cleanliness_status"] = cleanliness_status
                det["cleanliness_confidence"] = cleanliness_confidence
                det["cleanliness_model_version"] = cleanliness_model_version
                det["requires_review"] = False
                det["grade"] = calculate_grade(
                    damage.status,
                    det.get("size"),
                    det["confidence"],
                    confidence,
                    cleanliness_status=cleanliness_status,
                    damage_confidence=damage.confidence,
                    damage_threshold=damage_review_threshold,
                )
                det["provisional_grade"] = provisional_grade(
                    damage.status,
                    det.get("size"),
                    damage_confidence=damage.confidence,
                    damage_threshold=damage_review_threshold,
                    detection_confidence=det["confidence"],
                    detection_threshold=confidence,
                )
                reasons = collect_review_reasons(
                    detection_confidence=det["confidence"],
                    detection_threshold=confidence,
                    damage_status=damage.status,
                    damage_confidence=damage.confidence,
                    damage_threshold=damage_review_threshold,
                    cleanliness_status=cleanliness_status,
                    cleanliness_confidence=cleanliness_confidence,
                    size=det.get("size"),
                    geometry_quality=det.get("geometry_quality"),
                    weight_g=det.get("weight_g"),
                    damage_review_reason=damage.review_reason,
                    tray_over_capacity=result.get("over_capacity", False),
                )
                det["requires_review"] = bool(reasons)
                det["review_reason"] = join_review_reasons(reasons)

            annotated_path = None
            if save_annotated and result["detections"]:
                annotated = self.predictor.draw_boxes(img, result["detections"])
                base_name = Path(prediction.file_path).stem
                ext = Path(prediction.file_path).suffix
                annotated_path = str(
                    Path(prediction.file_path).parent
                    / f"annotated_{base_name}{ext}"
                )
                if not cv2.imwrite(annotated_path, annotated):
                    annotated_path = None

            prediction.inference_metadata = self._inference_metadata(
                confidence, result
            )

            for det in result["detections"]:
                box = DetectionBox(
                    prediction_id=prediction.id,
                    egg_id=det.get("egg_id"),
                    class_name=det["class"],
                    confidence=det["confidence"],
                    x1=det["bbox"][0],
                    y1=det["bbox"][1],
                    x2=det["bbox"][2],
                    y2=det["bbox"][3],
                    annotated=annotated_path is not None,
                    size=det.get("size"),
                    length_mm=det.get("length_mm"),
                    breadth_mm=det.get("breadth_mm"),
                    geometry_quality=det.get("geometry_quality"),
                    weight_g=det.get("weight_g"),
                    grade=det.get("grade"),
                    provisional_grade=det.get("provisional_grade"),
                    requires_review=bool(det.get("requires_review")),
                    review_reason=det.get("review_reason"),
                    damage_status=det.get("damage_status"),
                    damage_confidence=det.get("damage_confidence"),
                    damage_model_version=det.get("damage_model_version"),
                    cleanliness_status=det.get("cleanliness_status"),
                    cleanliness_confidence=det.get("cleanliness_confidence"),
                    cleanliness_model_version=det.get(
                        "cleanliness_model_version"
                    ),
                    detection_model_version=det.get(
                        "detection_model_version"
                    ),
                    calibration_version=det.get("calibration_version"),
                )
                db.add(box)

            prediction.status = "completed"
            prediction.result_json = result
            prediction.annotated_path = annotated_path
            prediction.total_detections = result["total"]
            prediction.damaged_count = sum(
                det.get("damage_status") == "Damaged"
                for det in result["detections"]
            )
            prediction.not_damaged_count = sum(
                det.get("damage_status") == "Not Damaged"
                for det in result["detections"]
            )
            prediction.needs_review_count = sum(
                bool(det.get("requires_review"))
                for det in result["detections"]
            )
            prediction.processing_time_ms = int((time.time() - start_time) * 1000)
            prediction.completed_at = datetime.utcnow()

            db.commit()
            db.refresh(prediction)

            try:
                append_prediction(prediction)
            except Exception as exc:
                prediction.error_message = f"CSV export failed: {exc}"
                db.commit()

        except Exception as e:
            prediction.status = "failed"
            prediction.error_message = str(e)
            db.commit()
            raise

        return prediction

    def _inference_metadata(self, confidence: float, result: dict) -> dict:
        damage_threshold = getattr(
            self.damage_classifier,
            "damaged_threshold",
            settings.DAMAGE_DECISION_THRESHOLD,
        )
        damage_review = getattr(
            self.damage_classifier,
            "review_threshold",
            settings.DAMAGE_REVIEW_THRESHOLD,
        )
        if self.cleanliness_classifier is None:
            cleanliness_threshold = None
            cleanliness_review = None
            cleanliness_model_version = None
        else:
            cleanliness_threshold = getattr(
                self.cleanliness_classifier,
                "stained_threshold",
                settings.CLEANLINESS_DECISION_THRESHOLD,
            )
            cleanliness_review = getattr(
                self.cleanliness_classifier,
                "review_threshold",
                settings.CLEANLINESS_REVIEW_THRESHOLD,
            )
            cleanliness_model_version = getattr(
                self.cleanliness_classifier, "model_version", None
            )
        return {
            "detector_model_version": getattr(
                self.predictor, "model_version", None
            ),
            "damage_model_version": getattr(
                self.damage_classifier, "model_version", None
            ),
            "cleanliness_model_version": cleanliness_model_version,
            "calibration_version": settings.CALIBRATION_VERSION,
            "detection_threshold": confidence,
            "damage_decision_threshold": damage_threshold,
            "damage_review_threshold": damage_review,
            "cleanliness_decision_threshold": cleanliness_threshold,
            "cleanliness_review_threshold": cleanliness_review,
            "crop_padding": settings.DETECTION_CROP_PADDING,
            "max_tray_eggs": result.get("max_eggs", settings.MAX_TRAY_EGGS),
            "tray_count_valid": result.get("count_valid", True),
            "tray_over_capacity": result.get("over_capacity", False),
            "tray_count_status": result.get("count_status"),
            "duplicate_boxes_removed": result.get(
                "duplicate_boxes_removed", 0
            ),
        }

    def get_predictions(
        self, db: Session, user_id: int, skip: int = 0, limit: int = 20
    ) -> list[Prediction]:
        return (
            db.query(Prediction)
            .filter(Prediction.user_id == user_id)
            .order_by(Prediction.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_prediction_detail(
        self, db: Session, prediction_id: int, user_id: int
    ) -> Optional[Prediction]:
        return (
            db.query(Prediction)
            .filter(
                Prediction.id == prediction_id,
                Prediction.user_id == user_id,
            )
            .first()
        )

    def delete_prediction(
        self, db: Session, prediction_id: int, user_id: int
    ) -> bool:
        prediction = self.get_prediction_detail(db, prediction_id, user_id)
        if not prediction:
            return False

        try:
            if os.path.exists(prediction.file_path):
                os.remove(prediction.file_path)
            if prediction.annotated_path and os.path.exists(
                prediction.annotated_path
            ):
                os.remove(prediction.annotated_path)
        except Exception:
            pass

        db.delete(prediction)
        db.commit()
        return True


prediction_service = PredictionService()


def crop_detection(
    image,
    bbox: list[int],
    padding_ratio: float,
    min_crop_size: int,
):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        return image[0:0, 0:0]
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    if right - left < min_crop_size or bottom - top < min_crop_size:
        return image[0:0, 0:0]
    return image[top:bottom, left:right]
