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


class DamageClassifier:
    """Inference wrapper for the binary damage CNN checkpoint."""

    EXPECTED_CLASSES = ("Damaged", "Not Damaged")

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        review_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path or settings.DAMAGE_MODEL_PATH)
        self.review_threshold = (
            review_threshold
            if review_threshold is not None
            else settings.DAMAGE_REVIEW_THRESHOLD
        )
        self.device_name = device
        self._model = None
        self._torch = None
        self._transform = None
        self.class_names = self.EXPECTED_CLASSES

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
        from torch import nn
        from torchvision import transforms

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

        class EggGradingCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 56 * 56, 128)
                self.fc2 = nn.Linear(128, 2)

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                return self.fc2(torch.relu(self.fc1(x)))

        model = EggGradingCNN()
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self._select_device(torch))
        model.eval()
        self._torch = torch
        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((checkpoint.get("img_size", 224),) * 2),
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
        confidence, class_index = self._torch.max(probabilities, dim=0)
        confidence_value = float(confidence.item())
        if confidence_value < self.review_threshold:
            return DamageResult(
                "Needs Review",
                confidence_value,
                self.model_version,
                "low_confidence",
            )
        return DamageResult(
            self.class_names[int(class_index.item())],
            confidence_value,
            self.model_version,
        )
