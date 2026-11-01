"""Deterministic per-egg grading and review policy.

The final grade requires damage, cleanliness, and size. If cleanliness is
``Unavailable``, the final grade is ``N/A``. ``provisional_grade`` exposes the
damage+size decision without pretending cleanliness was evaluated. Size-derived
letter grades additionally require a calibrated camera
(``settings.CALIBRATION_VERSION``); ``Reject`` never depends on calibration.
The current cleanliness checkpoint is a web/stock candidate and is not
release-approved.
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
    *,
    size_calibrated: bool = True,
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
    if not size_calibrated:
        return "N/A"
    return SIZE_GRADES.get(normalize_size(size), "N/A")


def map_to_grade(
    damage_status: Optional[str],
    cleanliness_status: Optional[str],
    size: Optional[str],
    *,
    size_calibrated: bool = True,
) -> str:
    """Final grade from all three criteria."""
    damage = normalize_damage_status(damage_status)
    cleanliness = normalize_cleanliness_status(cleanliness_status)
    if damage == "Damaged" or cleanliness == "Stained":
        return "Reject"
    if damage != "Not Damaged" or cleanliness != "Clean":
        return "N/A"
    if not size_calibrated:
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
    weight_g: Optional[float],
    weight_calibrated: bool = False,
    size_calibrated: bool = True,
    damage_review_reason: Optional[str] = None,
    tray_over_capacity: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if (
        detection_confidence is not None
        and detection_confidence < detection_threshold
    ):
        reasons.append("low_detection_confidence")
    damage = normalize_damage_status(damage_status)
    if damage == "Needs Review":
        reasons.append("damage_needs_review")
    elif (
        damage == "Not Damaged"
        and damage_confidence is not None
        and damage_confidence < damage_threshold
    ):
        # Only the not-damaged confidence (1 - p) is a review signal here: a
        # damaged call already yields Reject, so flagging it again would only
        # inflate the review queue.
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

    if size_calibrated and normalize_size(size) == "Unknown":
        reasons.append("size_unknown")
    # Calibration is a tray-level configuration state surfaced in inference
    # metadata and the UI, not a per-egg defect.
    if weight_calibrated and weight_g is None:
        reasons.append("weight_unavailable")
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
    size_calibrated: bool = True,
) -> str:
    """Compatibility wrapper; cleanliness defaults to unavailable, not clean."""
    if detection_confidence < detection_threshold:
        return "N/A"
    if normalize_damage_status(damage_status) == "Damaged":
        return "Reject"
    if damage_confidence is not None and damage_confidence < damage_threshold:
        return "N/A"
    return map_to_grade(
        damage_status,
        cleanliness_status,
        size,
        size_calibrated=size_calibrated,
    )
