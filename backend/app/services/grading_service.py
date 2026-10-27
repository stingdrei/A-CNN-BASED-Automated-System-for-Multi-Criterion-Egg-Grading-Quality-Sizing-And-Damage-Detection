"""Deterministic per-egg grading and review policy.

The final grade requires damage, cleanliness, and size. Cleanliness is not
implemented yet, so final grades are ``N/A``. ``provisional_grade`` exposes the
damage+size decision without pretending cleanliness was evaluated.
"""

from typing import Iterable, Optional


FINAL_GRADES = ("A", "B", "C", "N/A", "Reject")
SIZE_GRADES = {"Large": "A", "Medium": "B", "Small": "C"}
CLEANLINESS_UNAVAILABLE = "Unavailable"


def normalize_size(size: Optional[str]) -> str:
    if not size:
        return "Unknown"
    normalized = str(size).strip().title()
    if normalized in {"Small", "Medium", "Large"}:
        return normalized
    return "Unknown"


def normalize_damage_status(status: Optional[str]) -> str:
    if status in {"Damaged", "Not Damaged", "Needs Review"}:
        return status
    return "Needs Review"


def normalize_cleanliness_status(status: Optional[str]) -> str:
    if status in {
        "Clean",
        "Stained",
        "Needs Review",
        "Unavailable",
        "Unknown",
    }:
        return status
    return "Unknown"


def provisional_grade(
    damage_status: Optional[str],
    size: Optional[str],
    damage_confidence: Optional[float] = None,
    damage_threshold: float = 0.75,
    detection_confidence: Optional[float] = None,
    detection_threshold: float = 0.75,
) -> str:
    """Grade from damage and size only; cleanliness is intentionally ignored."""
    status = normalize_damage_status(damage_status)
    if detection_confidence is not None and detection_confidence < detection_threshold:
        return "N/A"
    if status == "Damaged":
        return "Reject"
    if status != "Not Damaged":
        return "N/A"
    if damage_confidence is not None and damage_confidence < damage_threshold:
        return "N/A"
    return SIZE_GRADES.get(normalize_size(size), "N/A")


def map_to_grade(
    damage_status: Optional[str],
    cleanliness_status: Optional[str],
    size: Optional[str],
) -> str:
    """Final grade from all three criteria."""
    damage = normalize_damage_status(damage_status)
    cleanliness = normalize_cleanliness_status(cleanliness_status)
    if damage == "Damaged" or cleanliness == "Stained":
        return "Reject"
    if damage != "Not Damaged" or cleanliness != "Clean":
        return "N/A"
    return SIZE_GRADES.get(normalize_size(size), "N/A")


def collect_review_reasons(
    *,
    detection_confidence: Optional[float],
    detection_threshold: float,
    damage_status: Optional[str],
    damage_confidence: Optional[float],
    damage_threshold: float,
    cleanliness_status: Optional[str],
    cleanliness_confidence: Optional[float] = None,
    cleanliness_threshold: float = 0.75,
    size: Optional[str],
    geometry_quality: Optional[str],
    weight_g: Optional[float],
    damage_review_reason: Optional[str] = None,
    tray_over_capacity: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if (
        detection_confidence is not None
        and detection_confidence < detection_threshold
    ):
        reasons.append("low_detection_confidence")
    if normalize_damage_status(damage_status) == "Needs Review":
        reasons.append("damage_needs_review")
    elif (
        damage_confidence is not None
        and damage_confidence < damage_threshold
    ):
        reasons.append("low_damage_confidence")
    if damage_review_reason:
        reasons.append(damage_review_reason)

    cleanliness = normalize_cleanliness_status(cleanliness_status)
    if cleanliness in {CLEANLINESS_UNAVAILABLE, "Unknown"}:
        reasons.append("cleanliness_unavailable")
    elif cleanliness == "Needs Review":
        reasons.append("cleanliness_needs_review")
    elif (
        cleanliness_confidence is not None
        and cleanliness_confidence < cleanliness_threshold
    ):
        reasons.append("low_cleanliness_confidence")

    if normalize_size(size) == "Unknown":
        reasons.append("size_unknown")
    elif geometry_quality == "bbox_fallback":
        reasons.append("size_uncalibrated")
    if weight_g is None:
        reasons.append("weight_uncalibrated")
    if tray_over_capacity:
        reasons.append("tray_over_capacity")

    return list(dict.fromkeys(reasons))


def join_review_reasons(reasons: Iterable[str]) -> Optional[str]:
    unique = list(dict.fromkeys(reason for reason in reasons if reason))
    return ";".join(unique) if unique else None


def calculate_grade(
    damage_status: Optional[str],
    size: Optional[str],
    detection_confidence: float,
    detection_threshold: float,
    *,
    cleanliness_status: Optional[str] = "Unavailable",
    damage_confidence: Optional[float] = None,
    damage_threshold: float = 0.75,
) -> str:
    """Compatibility wrapper; cleanliness defaults to unavailable, not clean."""
    if detection_confidence < detection_threshold:
        return "N/A"
    if normalize_damage_status(damage_status) == "Damaged":
        return "Reject"
    if damage_confidence is not None and damage_confidence < damage_threshold:
        return "N/A"
    return map_to_grade(damage_status, cleanliness_status, size)
