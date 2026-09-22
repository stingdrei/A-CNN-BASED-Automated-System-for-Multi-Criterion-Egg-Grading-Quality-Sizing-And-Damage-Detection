"""Camera-calibrated geometry helpers for per-egg measurements."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class EggGeometry:
    length_px: float
    breadth_px: float
    length_mm: float
    breadth_mm: float
    quality: str


def fit_ellipse_axes(mask: np.ndarray, mm_per_pixel: float) -> Optional[EggGeometry]:
    """Fit an ellipse to a binary egg mask and return major/minor axes.

    OpenCV returns ellipse diameters, so the values are already full-axis
    lengths. A valid fit needs at least five foreground points.
    """
    if mm_per_pixel <= 0 or mask is None:
        return None
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5 or cv2.contourArea(contour) <= 0:
        return None
    (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
    length_px, breadth_px = sorted((float(axis_a), float(axis_b)), reverse=True)
    return EggGeometry(
        length_px=length_px,
        breadth_px=breadth_px,
        length_mm=length_px * mm_per_pixel,
        breadth_mm=breadth_px * mm_per_pixel,
        quality="ellipse_fit",
    )


def bbox_geometry(
    width_px: int, height_px: int, mm_per_pixel: float
) -> Optional[EggGeometry]:
    """Fallback geometry for detection-only mode; mark it as lower quality."""
    if width_px <= 0 or height_px <= 0 or mm_per_pixel <= 0:
        return None
    length_px, breadth_px = sorted(
        (float(width_px), float(height_px)), reverse=True
    )
    return EggGeometry(
        length_px=length_px,
        breadth_px=breadth_px,
        length_mm=length_px * mm_per_pixel,
        breadth_mm=breadth_px * mm_per_pixel,
        quality="bbox_fallback",
    )
