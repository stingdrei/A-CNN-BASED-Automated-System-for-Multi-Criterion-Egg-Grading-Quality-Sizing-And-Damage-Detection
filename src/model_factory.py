"""Single source of truth for classifier architectures.

Training, evaluation, and backend inference resolve architecture names through
this module so checkpoints can carry enough metadata to be rebuilt faithfully.
"""

import warnings

import torch.nn as nn

try:  # top-level import when src/ is on sys.path
    from model import EggGradingCNN
except ModuleNotFoundError:  # package import from the repository root
    from .model import EggGradingCNN

ARCH_CUSTOM = "custom"
ARCH_RESNET18 = "resnet18"
SUPPORTED_ARCHS = (ARCH_CUSTOM, ARCH_RESNET18)


def _build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision import models

    weights = None
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT
        except (ImportError, AttributeError):
            weights = None

    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:  # offline / weight-download failure
        warnings.warn(
            f"Could not load pretrained resnet18 weights ({exc}); "
            "falling back to random initialization.",
            UserWarning,
            stacklevel=2,
        )
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(
    arch: str, num_classes: int = 2, pretrained: bool = False
) -> nn.Module:
    """Build a classifier by architecture name.

    Supported: ``custom`` (EggGradingCNN) and ``resnet18`` (torchvision).
    """
    if arch == ARCH_CUSTOM:
        return EggGradingCNN(num_classes=num_classes)
    if arch == ARCH_RESNET18:
        return _build_resnet18(num_classes, pretrained)
    raise ValueError(
        f"Unknown architecture {arch!r}; expected one of {SUPPORTED_ARCHS}"
    )


def checkpoint_metadata(
    arch: str, num_classes: int = 2, pretrained: bool = False, **extra
) -> dict:
    """Metadata to persist alongside a checkpoint's state_dict."""
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Unknown architecture {arch!r}; expected one of {SUPPORTED_ARCHS}"
        )
    metadata = {
        "arch": arch,
        "arch_args": {"num_classes": num_classes, "pretrained": pretrained},
    }
    metadata.update(extra)
    return metadata
