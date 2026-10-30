import os
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2

from app.config import settings
from src.grading import estimate_weight as estimate_egg_weight
from src.grading import size_category as classify_egg_size
from src.geometry import bbox_geometry
from src.tray_count import deduplicate_boxes, tray_count_status


class YOLOPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.75,
        mm_per_pixel: Optional[float] = None,
        weight_k: Optional[float] = None,
        max_eggs: Optional[int] = None,
        duplicate_iou: Optional[float] = None,
    ):
        self.model_path = model_path or settings.MODEL_PATH
        self.confidence = confidence
        self.mm_per_pixel = mm_per_pixel or settings.MM_PER_PIXEL
        self.weight_k = settings.WEIGHT_K if weight_k is None else weight_k
        self.max_eggs = settings.MAX_TRAY_EGGS if max_eggs is None else max_eggs
        self.duplicate_iou = (
            settings.DETECTION_DUPLICATE_IOU
            if duplicate_iou is None
            else duplicate_iou
        )
        if self.max_eggs < 1:
            raise ValueError("max_eggs must be at least 1")
        if not 0 < self.duplicate_iou <= 1:
            raise ValueError("duplicate_iou must be in (0, 1]")
        self.class_names = ["egg"]
        self._model = None
        # Ultralytics predictors keep internal state; serialize model calls so
        # concurrent inspections cannot corrupt each other. Per-request
        # thresholds and calibration are passed as arguments, never stored.
        self._inference_lock = threading.Lock()

    @property
    def model_version(self) -> str:
        return Path(self.model_path).name

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            if hasattr(self._model, "names") and self._model.names:
                self.class_names = self._model.names
        return self._model

    def get_class_name(self, class_id: int) -> str:
        if isinstance(self.class_names, dict):
            return self.class_names.get(class_id, f"class_{class_id}")
        elif isinstance(self.class_names, list) and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return "egg"

    def load_model(self) -> bool:
        if not os.path.exists(self.model_path):
            print(f"Warning: Model file not found at {self.model_path}")
            return False
        try:
            _ = self.model
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_image(
        self,
        image_path: str,
        confidence: Optional[float] = None,
        mm_per_pixel: Optional[float] = None,
    ) -> Dict[str, Any]:
        effective_confidence = (
            self.confidence if confidence is None else confidence
        )
        with self._inference_lock:
            results = self.model(image_path, conf=effective_confidence)
        return self._parse_results(results[0], mm_per_pixel=mm_per_pixel)

    def predict_frame(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        mm_per_pixel: Optional[float] = None,
    ) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            raise ValueError("Image frame is empty")
        effective_confidence = (
            self.confidence if confidence is None else confidence
        )
        with self._inference_lock:
            results = self.model(frame, conf=effective_confidence)
        return self._parse_results(results[0], mm_per_pixel=mm_per_pixel)

    def predict_bytes(
        self,
        image_bytes: bytes,
        confidence: Optional[float] = None,
        mm_per_pixel: Optional[float] = None,
    ) -> Dict[str, Any]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image bytes")
        return self.predict_frame(
            frame, confidence=confidence, mm_per_pixel=mm_per_pixel
        )

    def _parse_results(
        self, result, mm_per_pixel: Optional[float] = None
    ) -> Dict[str, Any]:
        effective_mm_per_pixel = (
            self.mm_per_pixel if mm_per_pixel is None else mm_per_pixel
        )
        detections = []
        duplicate_boxes_removed = 0
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            candidates = []
            for index in range(len(xyxy)):
                box = xyxy[index]
                x1, y1, x2, y2 = [int(x) for x in box]
                if x2 <= x1 or y2 <= y1:
                    continue
                candidates.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(confs[index]),
                        "class_id": int(clss[index]),
                    }
                )
            candidates, duplicate_boxes_removed = deduplicate_boxes(
                candidates, self.duplicate_iou
            )

            for egg_id, candidate in enumerate(candidates, start=1):
                x1, y1, x2, y2 = candidate["bbox"]
                geometry = bbox_geometry(
                    x2 - x1, y2 - y1, effective_mm_per_pixel
                )
                if geometry is None:
                    size_cat = "Unknown"
                    length_mm = None
                    breadth_mm = None
                    geometry_quality = "unknown"
                    weight = None
                else:
                    size_cat = classify_egg_size(geometry.length_mm)
                    length_mm = round(geometry.length_mm, 2)
                    breadth_mm = round(geometry.breadth_mm, 2)
                    geometry_quality = geometry.quality
                    weight = estimate_egg_weight(
                        geometry.length_mm,
                        geometry.breadth_mm,
                        self.weight_k,
                    )
                class_name = self.get_class_name(candidate["class_id"])
                detections.append({
                    "egg_id": egg_id,
                    "class": class_name,
                    "confidence": candidate["confidence"],
                    "bbox": [x1, y1, x2, y2],
                    "size": size_cat,
                    "length_mm": length_mm,
                    "breadth_mm": breadth_mm,
                    "geometry_quality": geometry_quality,
                    "weight_g": weight,
                    "grade": "N/A",
                    "provisional_grade": "N/A",
                    "detection_model_version": self.model_version,
                    "calibration_version": settings.CALIBRATION_VERSION,
                })

        count = tray_count_status(len(detections), self.max_eggs)
        return {
            "detections": detections,
            **count,
            "duplicate_boxes_removed": duplicate_boxes_removed,
            "damaged": 0,
            "not_damaged": 0,
        }

    def draw_boxes(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        img = image.copy()

        grade_colors = {
            "A": (34, 197, 94),
            "B": (59, 130, 246),
            "C": (245, 158, 11),
            "N/A": (107, 114, 128),
            "Reject": (239, 68, 68),
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            final_grade = det.get("grade", "N/A")
            provisional = det.get("provisional_grade", "N/A")
            if final_grade == "N/A" and provisional not in (None, "N/A"):
                grade = provisional
                label = f"P:{grade} {conf:.2f}"
            else:
                grade = final_grade
                label = f"{grade} {conf:.2f}"
            weight = det.get("weight_g")
            if weight:
                label += f" {weight}g"

            color = grade_colors.get(grade, (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return img
