import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2

from app.config import settings


class YOLOPredictor:
    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.75, mm_per_pixel: Optional[float] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.confidence = confidence
        self.mm_per_pixel = mm_per_pixel or settings.MM_PER_PIXEL
        self.class_names = ["egg"]
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

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

    def calculate_size_category(self, diameter_px: int) -> str:
        if diameter_px <= 0:
            return "unknown"
        diameter_mm = diameter_px * self.mm_per_pixel
        if diameter_mm < 50:
            return "small"
        elif diameter_mm < 60:
            return "medium"
        else:
            return "large"

    def estimate_weight(self, diameter_px: int) -> Optional[float]:
        if diameter_px <= 0:
            return None
        diameter_mm = diameter_px * self.mm_per_pixel
        weight = 0.0005 * (diameter_mm ** 3)
        return round(max(30, min(80, weight)), 1)

    def map_to_grade(self, class_name: str, size_category: Optional[str]) -> str:
        if not size_category or size_category == "unknown":
            return "N/A"
        if size_category == "large":
            return "A"
        elif size_category == "medium":
            return "B"
        else:
            return "C"

    def predict_image(self, image_path: str) -> Dict[str, Any]:
        results = self.model(image_path, conf=self.confidence)
        return self._parse_results(results[0])

    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        results = self.model(frame, conf=self.confidence)
        return self._parse_results(results[0])

    def predict_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self.predict_frame(frame)

    def _parse_results(self, result) -> Dict[str, Any]:
        detections = []
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = [int(x) for x in box]
                diameter_px = max(x2 - x1, y2 - y1)
                size_cat = self.calculate_size_category(diameter_px)
                weight = self.estimate_weight(diameter_px)
                grade = self.map_to_grade(self.class_names[int(clss[i])], size_cat)
                
                detections.append({
                    "class": self.class_names[int(clss[i])],
                    "confidence": float(confs[i]),
                    "bbox": [x1, y1, x2, y2],
                    "size_category": size_cat,
                    "weight_g": weight,
                    "grade": grade
                })

        return {
            "detections": detections,
            "total": len(detections),
            # Damage is intentionally unknown until the damage classifier runs.
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
            cls = det["class"]
            conf = det["confidence"]
            grade = det.get("grade", "N/A")
            weight = det.get("weight_g", 0)

            color = grade_colors.get(grade, (128, 128, 128))
            label = f"{grade} {conf:.2f}"
            if weight:
                label += f" {weight}g"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img