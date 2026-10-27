from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.config import settings


@dataclass(frozen=True)
class DamageResult:
    status: str
    confidence: float
    model_version: str
    review_reason: Optional[str] = None


def decide_damage(
    damaged_probability: float,
    damaged_threshold: float,
    review_threshold: float,
) -> tuple[str, float, Optional[str]]:
    """Map P(Damaged) to status, confidence, and review reason.

    A damaged prediction is never downgraded to review: the decision threshold
    is selected on validation to keep damaged recall high. Review applies only
    when a not-damaged prediction is below the confidence threshold.
    """
    if damaged_probability >= damaged_threshold:
        return "Damaged", damaged_probability, None
    confidence = 1.0 - damaged_probability
    if confidence < review_threshold:
        return "Needs Review", confidence, "low_confidence"
    return "Not Damaged", confidence, None


class DamageClassifier:
    """Inference wrapper for the binary damage CNN checkpoint."""

    EXPECTED_CLASSES = ("Damaged", "Not Damaged")

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        review_threshold: Optional[float] = None,
        damaged_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path or settings.DAMAGE_MODEL_PATH)
        self.review_threshold = (
            review_threshold
            if review_threshold is not None
            else settings.DAMAGE_REVIEW_THRESHOLD
        )
        self.damaged_threshold = (
            damaged_threshold
            if damaged_threshold is not None
            else settings.DAMAGE_DECISION_THRESHOLD
        )
        self._explicit_damaged_threshold = damaged_threshold is not None
        self.device_name = device
        self._model = None
        self._torch = None
        self._transform = None
        self.class_names = self.EXPECTED_CLASSES
        self.arch = "unknown"
        self.img_size = 224

    @property
    def model_version(self) -> str:
        return self.checkpoint_path.name

    def load_model(self) -> bool:
        if not self.checkpoint_path.is_file():
            return False
        try:
            self._ensure_model()
            return True
        except (ImportError, KeyError, RuntimeError, ValueError, OSError):
            self._model = None
            return False

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from torchvision import transforms

        from src.model_factory import build_model

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self._select_device(torch),
            weights_only=False,
        )
        class_names = tuple(checkpoint.get("class_names", ()))
        if class_names != self.EXPECTED_CLASSES:
            raise ValueError(
                f"Unexpected damage checkpoint classes: {class_names!r}"
            )

        arch = str(checkpoint.get("arch", "custom"))
        arch_args = checkpoint.get("arch_args") or {}
        num_classes = int(arch_args.get("num_classes", len(class_names)))
        if num_classes != len(class_names):
            raise ValueError(
                f"Checkpoint num_classes={num_classes} does not match "
                f"class_names={class_names!r}"
            )
        img_size = int(checkpoint.get("img_size", 224))
        if img_size <= 0:
            raise ValueError(f"Invalid checkpoint img_size: {img_size}")
        if not self._explicit_damaged_threshold:
            checkpoint_threshold = checkpoint.get("damaged_threshold")
            if checkpoint_threshold is not None:
                checkpoint_threshold = float(checkpoint_threshold)
                if not 0 < checkpoint_threshold < 1:
                    raise ValueError(
                        "Invalid checkpoint damaged_threshold: "
                        f"{checkpoint_threshold}"
                    )
                self.damaged_threshold = checkpoint_threshold

        model = build_model(
            arch,
            num_classes=num_classes,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self._select_device(torch))
        model.eval()
        self._torch = torch
        self._model = model
        self.arch = arch
        self.img_size = img_size
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size,) * 2),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def _select_device(self, torch_module: Any):
        if self.device_name:
            return torch_module.device(self.device_name)
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    def classify_crop(self, crop: np.ndarray) -> DamageResult:
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return DamageResult(
                "Needs Review", 0.0, self.model_version, "empty_crop"
            )
        if crop.ndim not in (2, 3) or crop.shape[0] < 2 or crop.shape[1] < 2:
            return DamageResult(
                "Needs Review", 0.0, self.model_version, "invalid_crop"
            )

        self._ensure_model()
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        else:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        tensor = self._transform(crop).unsqueeze(0).to(self._model.parameters().__next__().device)
        with self._torch.no_grad():
            probabilities = self._torch.softmax(self._model(tensor), dim=1)[0]
        damaged_probability = float(probabilities[0].item())
        status, confidence, review_reason = decide_damage(
            damaged_probability,
            self.damaged_threshold,
            self.review_threshold,
        )
        return DamageResult(
            status,
            confidence,
            self.model_version,
            review_reason,
        )
