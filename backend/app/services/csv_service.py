"""Append per-egg rows to the canonical CSV export."""

import csv
from pathlib import Path
from typing import Optional

from app.config import settings


CSV_FIELDS = [
    "timestamp",
    "tray_id",
    "prediction_id",
    "egg_id",
    "file_name",
    "class_name",
    "detection_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "size",
    "length_mm",
    "breadth_mm",
    "geometry_quality",
    "weight_g",
    "damage_status",
    "damage_confidence",
    "cleanliness_status",
    "cleanliness_confidence",
    "grade",
    "provisional_grade",
    "requires_review",
    "review_reason",
    "detector_model_version",
    "damage_model_version",
    "cleanliness_model_version",
    "calibration_version",
]


def rows_for_prediction(prediction) -> list[dict]:
    timestamp = prediction.captured_at or prediction.created_at
    rows = []
    for box in prediction.detection_boxes:
        rows.append(
            {
                "timestamp": timestamp.isoformat() if timestamp else "",
                "tray_id": prediction.tray_id or "",
                "prediction_id": prediction.id,
                "egg_id": box.egg_id if box.egg_id is not None else "",
                "file_name": prediction.file_name,
                "class_name": box.class_name,
                "detection_confidence": box.confidence,
                "x1": box.x1,
                "y1": box.y1,
                "x2": box.x2,
                "y2": box.y2,
                "size": box.size or "Unknown",
                "length_mm": box.length_mm if box.length_mm is not None else "",
                "breadth_mm": box.breadth_mm if box.breadth_mm is not None else "",
                "geometry_quality": box.geometry_quality or "unknown",
                "weight_g": box.weight_g if box.weight_g is not None else "",
                "damage_status": box.damage_status or "Needs Review",
                "damage_confidence": (
                    box.damage_confidence
                    if box.damage_confidence is not None
                    else ""
                ),
                "cleanliness_status": box.cleanliness_status or "Unknown",
                "cleanliness_confidence": (
                    box.cleanliness_confidence
                    if box.cleanliness_confidence is not None
                    else ""
                ),
                "grade": box.grade or "N/A",
                "provisional_grade": box.provisional_grade or "N/A",
                "requires_review": (
                    "true" if box.requires_review else "false"
                ),
                "review_reason": box.review_reason or "",
                "detector_model_version": box.detection_model_version or "",
                "damage_model_version": box.damage_model_version or "",
                "cleanliness_model_version": box.cleanliness_model_version or "",
                "calibration_version": box.calibration_version or "",
            }
        )
    return rows


def append_prediction(prediction, path: Optional[str] = None) -> Path:
    destination = Path(path or settings.EGG_CSV_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not destination.is_file() or destination.stat().st_size == 0
    rows = rows_for_prediction(prediction)
    if not rows:
        return destination
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
    return destination
