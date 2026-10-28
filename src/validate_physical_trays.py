"""Validate physical 0-5 egg trays with the production detector.

This is a no-training release check. It runs the same YOLO predictor used by
the backend, compares predicted tray counts with expected counts, and writes
JSON/CSV evidence.

Expectations CSV::

    filename,expected_count,notes
    0_egg_tray.jpg,0,empty tray
    5_egg_tray.jpg,5,full tray

The release gate requires at least one 0-egg and one 5-egg tray, exact count
accuracy >= 0.95, <= 0.10 false positives per tray, and <= 0.02 missed eggs per
tray. Over-capacity expectations are reported but fail the 0-5 gate.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_expectations(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty expectations CSV: {path}")
    required = {"filename", "expected_count"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Expectations missing columns: {sorted(missing)}")

    parsed = []
    seen = set()
    for line_number, row in enumerate(rows, start=2):
        filename = (row.get("filename") or "").strip()
        if not filename:
            raise ValueError(f"Missing filename on line {line_number}")
        basename = Path(filename).name
        if basename in seen:
            raise ValueError(f"Duplicate image basename: {basename}")
        seen.add(basename)
        try:
            expected = int(row["expected_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid expected_count on line {line_number}: "
                f"{row['expected_count']!r}"
            ) from exc
        if expected < 0:
            raise ValueError(
                f"Negative expected_count on line {line_number}: {expected}"
            )
        parsed.append(
            {
                "filename": filename,
                "basename": basename,
                "expected_count": expected,
                "notes": (row.get("notes") or "").strip(),
            }
        )
    return parsed


def resolve_images(images_dir: Path, expectations: List[dict]) -> List[dict]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images_dir}")
    index: Dict[str, Path] = {}
    duplicates = []
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name in index:
            duplicates.append(path.name)
            continue
        index[path.name] = path
    if duplicates:
        raise ValueError(f"Duplicate image basenames in input: {sorted(set(duplicates))}")

    missing = [
        row["filename"]
        for row in expectations
        if row["basename"] not in index
    ]
    if missing:
        raise FileNotFoundError(f"Missing expected images: {missing}")

    resolved = []
    for row in expectations:
        resolved.append({**row, "image_path": index[row["basename"]]})
    return resolved


def evaluate_trays(
    expectations: List[dict],
    predictor: Any,
    max_eggs: int = 5,
    min_exact_accuracy: float = 0.95,
    max_false_positives_per_tray: float = 0.10,
    max_missed_eggs_per_tray: float = 0.02,
) -> Tuple[List[dict], dict]:
    results = []
    false_positives = 0
    missed = 0
    exact = 0
    errors = 0
    expected_empty = 0
    expected_five = 0
    empty_pass = 0
    five_pass = 0
    predicted_over_capacity = 0
    duplicate_boxes_removed = 0

    for row in expectations:
        expected = int(row["expected_count"])
        result = {
            "filename": row["filename"],
            "image_path": str(row["image_path"]),
            "expected_count": expected,
            "predicted_count": None,
            "exact": False,
            "false_positives": 0,
            "missed": 0,
            "over_capacity": False,
            "duplicate_boxes_removed": 0,
            "notes": row.get("notes", ""),
            "error": None,
        }
        try:
            prediction = predictor.predict_image(str(row["image_path"]))
            predicted = int(
                prediction.get("total", len(prediction.get("detections", [])))
            )
        except Exception as exc:  # Keep per-image evidence instead of aborting.
            errors += 1
            result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
            continue

        fp = max(0, predicted - expected)
        miss = max(0, expected - predicted)
        over = bool(prediction.get("over_capacity", predicted > max_eggs))
        removed = int(prediction.get("duplicate_boxes_removed", 0))

        result.update(
            {
                "predicted_count": predicted,
                "exact": predicted == expected,
                "false_positives": fp,
                "missed": miss,
                "over_capacity": over,
                "duplicate_boxes_removed": removed,
            }
        )
        results.append(result)

        exact += int(predicted == expected)
        false_positives += fp
        missed += miss
        duplicate_boxes_removed += removed
        predicted_over_capacity += int(over)
        if expected == 0:
            expected_empty += 1
            empty_pass += int(predicted == 0)
        if expected == max_eggs:
            expected_five += 1
            five_pass += int(predicted == max_eggs)

    evaluated = len(results) - errors
    exact_accuracy = exact / evaluated if evaluated else 0.0
    fp_per_tray = false_positives / evaluated if evaluated else 0.0
    missed_per_tray = missed / evaluated if evaluated else 0.0
    within_range = all(0 <= int(row["expected_count"]) <= max_eggs for row in expectations)

    gates = {
        "no_prediction_errors": {
            "value": errors,
            "target": 0,
            "pass": errors == 0,
        },
        "exact_count_accuracy": {
            "value": exact_accuracy,
            "target": min_exact_accuracy,
            "pass": exact_accuracy >= min_exact_accuracy,
        },
        "false_positives_per_tray": {
            "value": fp_per_tray,
            "target": max_false_positives_per_tray,
            "pass": fp_per_tray <= max_false_positives_per_tray,
        },
        "missed_eggs_per_tray": {
            "value": missed_per_tray,
            "target": max_missed_eggs_per_tray,
            "pass": missed_per_tray <= max_missed_eggs_per_tray,
        },
        "zero_egg_trays_present": {
            "value": expected_empty,
            "target": 1,
            "pass": expected_empty >= 1,
        },
        "zero_egg_trays_pass": {
            "value": empty_pass,
            "target": expected_empty,
            "pass": expected_empty > 0 and empty_pass == expected_empty,
        },
        "five_egg_trays_present": {
            "value": expected_five,
            "target": 1,
            "pass": expected_five >= 1,
        },
        "five_egg_trays_pass": {
            "value": five_pass,
            "target": expected_five,
            "pass": expected_five > 0 and five_pass == expected_five,
        },
        "expectations_within_zero_to_max": {
            "value": within_range,
            "target": True,
            "pass": within_range,
        },
    }
    summary = {
        "images": len(results),
        "evaluated": evaluated,
        "errors": errors,
        "exact_matches": exact,
        "exact_count_accuracy": exact_accuracy,
        "false_positives": false_positives,
        "false_positives_per_tray": fp_per_tray,
        "missed_eggs": missed,
        "missed_eggs_per_tray": missed_per_tray,
        "expected_empty_trays": expected_empty,
        "expected_five_egg_trays": expected_five,
        "predicted_over_capacity_trays": predicted_over_capacity,
        "duplicate_boxes_removed": duplicate_boxes_removed,
        "gates": gates,
        "release_ready": all(gate["pass"] for gate in gates.values()),
    }
    return results, summary


def write_csv(results: List[dict], path: Path) -> None:
    fields = [
        "filename",
        "expected_count",
        "predicted_count",
        "exact",
        "false_positives",
        "missed",
        "over_capacity",
        "duplicate_boxes_removed",
        "notes",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fields})


def build_predictor(args: argparse.Namespace) -> Any:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.ml.yolo_inference import YOLOPredictor

    return YOLOPredictor(
        model_path=args.model,
        confidence=args.confidence,
        max_eggs=args.max_eggs,
        duplicate_iou=args.duplicate_iou,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="egg_detection/finetune_egg_v1/weights/best.pt",
    )
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--max-eggs", type=int, default=5)
    parser.add_argument("--duplicate-iou", type=float, default=0.85)
    parser.add_argument(
        "--min-exact-accuracy", type=float, default=0.95
    )
    parser.add_argument(
        "--max-false-positives-per-tray", type=float, default=0.10
    )
    parser.add_argument(
        "--max-missed-eggs-per-tray", type=float, default=0.02
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/physical_tray_validation.json")
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("reports/physical_tray_validation.csv")
    )
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    if args.max_eggs < 1:
        raise ValueError("--max-eggs must be positive")
    if not 0 <= args.min_exact_accuracy <= 1:
        raise ValueError("--min-exact-accuracy must be in [0, 1]")
    if args.max_false_positives_per_tray < 0:
        raise ValueError("--max-false-positives-per-tray must be non-negative")
    if args.max_missed_eggs_per_tray < 0:
        raise ValueError("--max-missed-eggs-per-tray must be non-negative")

    expectations = resolve_images(
        args.images, read_expectations(args.expectations)
    )
    predictor = build_predictor(args)
    results, summary = evaluate_trays(
        expectations,
        predictor,
        max_eggs=args.max_eggs,
        min_exact_accuracy=args.min_exact_accuracy,
        max_false_positives_per_tray=args.max_false_positives_per_tray,
        max_missed_eggs_per_tray=args.max_missed_eggs_per_tray,
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "confidence": args.confidence,
        "max_eggs": args.max_eggs,
        "duplicate_iou": args.duplicate_iou,
        "expectations": str(args.expectations),
        "images_dir": str(args.images),
        "summary": summary,
        "results": results,
        "limitations": [
            "Physical images and expected counts are operator-supplied.",
            "This validates detector tray counts only, not damage/cleanliness/size/weight.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(results, args.csv)

    print(json.dumps(summary, indent=2))
    print(f"report: {args.output}")
    print(f"csv: {args.csv}")
    if args.fail_on_gate and not summary["release_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
