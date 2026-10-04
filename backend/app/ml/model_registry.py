from typing import Optional

from app.config import settings
from app.ml.damage_inference import DamageClassifier
from app.ml.yolo_inference import YOLOPredictor


class ModelRegistry:
    def __init__(self) -> None:
        self.detector: Optional[YOLOPredictor] = None
        self.damage: Optional[DamageClassifier] = None
        self.detector_loaded = False
        self.damage_loaded = False

    def load(self) -> tuple[bool, bool]:
        self.detector = YOLOPredictor()
        detector_loaded = self.detector.load_model()
        self.damage = DamageClassifier()
        damage_loaded = self.damage.load_model()
        self.detector_loaded = detector_loaded
        self.damage_loaded = damage_loaded
        return detector_loaded, damage_loaded


model_registry = ModelRegistry()
