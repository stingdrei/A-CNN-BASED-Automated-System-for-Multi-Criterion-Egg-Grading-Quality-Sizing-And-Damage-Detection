import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2

from app.config import settings
from src.grading import estimate_weight as estimate_egg_weight
from src.grading import map_to_grade as map_egg_grade
from src.grading import size_category as classify_egg_size
from src.geometry import bbox_geometry


class YOLOPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.75,
        mm_per_pixel: Optional[float] = None,
        weight_k: Optional[float] = None,
    ):
        self.model_path = model_path or settings.MODEL_PATH
        self.confidence = confidence
        self.mm_per_pixel = mm_per_pixel or settings.MM_PER_PIXEL
        self.weight_k = settings.WEIGHT_K if weight_k is None else weight_k
        self.class_names = ["egg"]
        self._model = None

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
                geometry = bbox_geometry(x2 - x1, y2 - y1, self.mm_per_pixel)
                if geometry is None:
                    size_cat = "unknown"
                    weight = None
                else:
                    size_cat = classify_egg_size(geometry.length_mm).lower()
                    weight = estimate_egg_weight(
                        geometry.length_mm,
                        geometry.breadth_mm,
                        self.weight_k,
                    )
                class_name = self.get_class_name(int(clss[i]))
                # Provisional: final grade is set after damage/cleanliness run.
                grade = map_egg_grade("Unknown", "Unknown", size_cat.capitalize())

                detections.append({
                    "class": class_name,
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