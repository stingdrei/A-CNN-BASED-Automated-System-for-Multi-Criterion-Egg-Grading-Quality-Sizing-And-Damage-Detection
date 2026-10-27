"""Static-tray count rules shared by inference and evaluation."""

from typing import Any, Iterable


DEFAULT_MAX_EGGS = 5
DEFAULT_DUPLICATE_IOU = 0.85


def box_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_boxes(
    candidates: Iterable[dict[str, Any]],
    iou_threshold: float = DEFAULT_DUPLICATE_IOU,
) -> tuple[list[dict[str, Any]], int]:
    kept = []
    removed = 0
    for candidate in sorted(
        candidates, key=lambda item: item["confidence"], reverse=True
    ):
        if any(
            box_iou(candidate["bbox"], existing["bbox"]) >= iou_threshold
            for existing in kept
        ):
            removed += 1
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return kept, removed


def tray_count_status(
    total: int,
    max_eggs: int = DEFAULT_MAX_EGGS,
) -> dict[str, Any]:
    over_capacity = total > max_eggs
    return {
        "total": total,
        "max_eggs": max_eggs,
        "count_valid": total <= max_eggs,
        "over_capacity": over_capacity,
        "count_status": (
            "empty" if total == 0 else "over_capacity" if over_capacity else "ok"
        ),
    }
