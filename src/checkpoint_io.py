"""Checkpoint loading with a guarded unsafe escape hatch.

``torch.load`` unpickles arbitrary objects unless ``weights_only=True`` is
used. Every shipped checkpoint is a plain dict of tensors and primitives, so
safe loading is the default. ``ALLOW_UNSAFE_CHECKPOINT=1`` exists only for a
legacy checkpoint that has not been re-saved yet.
"""

import os
from typing import Any

UNSAFE_ENV_VAR = "ALLOW_UNSAFE_CHECKPOINT"
_TRUTHY = {"1", "true", "yes", "on"}


def allow_unsafe_checkpoint() -> bool:
    return os.environ.get(UNSAFE_ENV_VAR, "").strip().lower() in _TRUTHY


def load_checkpoint(path, *, map_location: Any = None):
    """Load a checkpoint with safe weights-only defaults."""
    import torch

    return torch.load(
        path,
        map_location=map_location,
        weights_only=not allow_unsafe_checkpoint(),
    )