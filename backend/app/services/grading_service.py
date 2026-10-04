from typing import Optional


def calculate_grade(
    damage_status: Optional[str],
    size_category: Optional[str],
    detection_confidence: float,
    detection_threshold: float,
) -> str:
    if detection_confidence < detection_threshold:
        return "Needs Review"
    if damage_status in (None, "Needs Review"):
        return "Needs Review"
    if damage_status == "Damaged":
        return "Reject"
    if size_category == "large":
        return "A"
    if size_category == "medium":
        return "B"
    if size_category == "small":
        return "C"
    return "Needs Review"
