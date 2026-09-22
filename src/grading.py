"""Pure, deterministic grading and measurement rules."""

from typing import Optional


SIZE_THRESHOLDS_MM = (53.0, 60.0)


def size_category(length_mm: float) -> str:
    if length_mm <= 0:
        return "Unknown"
    if length_mm < SIZE_THRESHOLDS_MM[0]:
        return "Small"
    if length_mm <= SIZE_THRESHOLDS_MM[1]:
        return "Medium"
    return "Large"


def estimate_weight(length_mm: float, breadth_mm: float, k: float) -> Optional[float]:
    if length_mm <= 0 or breadth_mm <= 0 or k <= 0:
        return None
    return round(k * length_mm * (breadth_mm ** 2), 1)


def map_to_grade(
    damage_status: str,
    cleanliness_status: str,
    size: str,
) -> str:
    if damage_status == "Damaged" or cleanliness_status == "Stained":
        return "Reject"
    if damage_status != "Not Damaged" or cleanliness_status != "Clean":
        return "N/A"
    return {"Large": "A", "Medium": "B", "Small": "C"}.get(size, "N/A")
