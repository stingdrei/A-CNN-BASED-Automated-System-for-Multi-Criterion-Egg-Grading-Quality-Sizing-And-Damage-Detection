import os
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session
import cv2

from app.models import Prediction, DetectionBox
from app.ml.yolo_inference import YOLOPredictor
from app.config import settings
from app.ml.damage_inference import DamageClassifier
from app.services.grading_service import calculate_grade


class PredictionService:
    def __init__(
        self,
        predictor: Optional[YOLOPredictor] = None,
        damage_classifier: Optional[DamageClassifier] = None,
    ):
        self.predictor = predictor or YOLOPredictor()
        self.damage_classifier = damage_classifier
    
    def create_prediction(
        self,
        db: Session,
        user_id: int,
        file_name: str,
        file_path: str,
        file_type: str
    ) -> Prediction:
        prediction = Prediction(
            user_id=user_id,
            file_name=file_name,
            file_type=file_type,
            file_path=file_path,
            status="processing"
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
        save_annotated: bool
    ) -> Prediction:
        start_time = time.time()
        
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise ValueError("Prediction not found")
        
        self.predictor.confidence = confidence
        
        try:
            if prediction.file_type == "image":
                img = cv2.imread(prediction.file_path)
                if img is None:
                    raise ValueError("Uploaded image could not be decoded")
                result = self.predictor.predict_image(prediction.file_path)
                if self.damage_classifier is None:
                    raise RuntimeError("Damage classifier is not configured")
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
                    det["review_reason"] = damage.review_reason
                    det["grade"] = calculate_grade(
                        damage.status,
                        det.get("size_category"),
                        det["confidence"],
                        confidence,
                    )
            else:
                raise ValueError("Video prediction is not supported yet")
            
            annotated_path = None
            if save_annotated and result["detections"]:
                if img is not None:
                    annotated = self.predictor.draw_boxes(img, result["detections"])
                    base_name = Path(prediction.file_path).stem
                    ext = Path(prediction.file_path).suffix
                    annotated_path = str(Path(prediction.file_path).parent / f"annotated_{base_name}{ext}")
                    if not cv2.imwrite(annotated_path, annotated):
                        annotated_path = None
            
            for det in result["detections"]:
                box = DetectionBox(
                    prediction_id=prediction.id,
                    class_name=det["class"],
                    confidence=det["confidence"],
                    x1=det["bbox"][0],
                    y1=det["bbox"][1],
                    x2=det["bbox"][2],
                    y2=det["bbox"][3],
                    annotated=annotated_path is not None,
                    size_category=det.get("size_category"),
                    weight_g=det.get("weight_g"),
                    grade=det.get("grade"),
                    damage_status=det.get("damage_status"),
                    damage_confidence=det.get("damage_confidence"),
                    damage_model_version=det.get("damage_model_version"),
                    review_reason=det.get("review_reason"),
                )
                db.add(box)
            
            prediction.status = "completed"
            prediction.result_json = result
            prediction.annotated_path = annotated_path
            prediction.total_detections = result["total"]
            prediction.damaged_count = sum(
                det.get("damage_status") == "Damaged" for det in result["detections"]
            )
            prediction.not_damaged_count = sum(
                det.get("damage_status") == "Not Damaged"
                for det in result["detections"]
            )
            prediction.needs_review_count = sum(
                det.get("damage_status") == "Needs Review"
                for det in result["detections"]
            )
            prediction.processing_time_ms = int((time.time() - start_time) * 1000)
            prediction.completed_at = datetime.utcnow()
            
            db.commit()
            db.refresh(prediction)
            
        except Exception as e:
            prediction.status = "failed"
            prediction.error_message = str(e)
            db.commit()
            raise
        
        return prediction
    
    def get_predictions(
        self,
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 20
    ) -> list[Prediction]:
        return db.query(Prediction).filter(
            Prediction.user_id == user_id
        ).order_by(
            Prediction.created_at.desc()
        ).offset(skip).limit(limit).all()
    
    def get_prediction_detail(
        self,
        db: Session,
        prediction_id: int,
        user_id: int
    ) -> Optional[Prediction]:
        return db.query(Prediction).filter(
            Prediction.id == prediction_id,
            Prediction.user_id == user_id
        ).first()
    
    def delete_prediction(
        self,
        db: Session,
        prediction_id: int,
        user_id: int
    ) -> bool:
        prediction = self.get_prediction_detail(db, prediction_id, user_id)
        if not prediction:
            return False
        
        try:
            if os.path.exists(prediction.file_path):
                os.remove(prediction.file_path)
            if prediction.annotated_path and os.path.exists(prediction.annotated_path):
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