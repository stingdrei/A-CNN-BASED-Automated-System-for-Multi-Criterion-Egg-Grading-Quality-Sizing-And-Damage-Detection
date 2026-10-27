"""Fit W = k * L * B^2 from weighed eggs.

Input CSV columns:
- length_mm
- breadth_mm
- actual_weight_g (or weight_g)

The fit has no intercept because zero axes must produce zero estimated weight.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"length_mm", "breadth_mm"}


def load_records(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    weight_column = (
        "actual_weight_g"
        if "actual_weight_g" in frame.columns
        else "weight_g"
    )
    missing = REQUIRED_COLUMNS | {weight_column}
    missing -= set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    length = pd.to_numeric(frame["length_mm"], errors="coerce").to_numpy(
        dtype=float
    )
    breadth = pd.to_numeric(frame["breadth_mm"], errors="coerce").to_numpy(
        dtype=float
    )
    actual = pd.to_numeric(frame[weight_column], errors="coerce").to_numpy(
        dtype=float
    )
    valid = (
        np.isfinite(length)
        & np.isfinite(breadth)
        & np.isfinite(actual)
        & (length > 0)
        & (breadth > 0)
        & (actual > 0)
    )
    if not valid.all():
        raise ValueError(
            f"{path} contains {int((~valid).sum())} invalid or non-positive rows"
        )
    return length, breadth, actual


def fit_calibration(
    length_mm: Iterable[float],
    breadth_mm: Iterable[float],
    actual_weight_g: Iterable[float],
) -> dict:
    length = np.asarray(list(length_mm), dtype=float)
    breadth = np.asarray(list(breadth_mm), dtype=float)
    actual = np.asarray(list(actual_weight_g), dtype=float)
    if not (len(length) == len(breadth) == len(actual)):
        raise ValueError("length, breadth, and weight arrays must match")
    if len(actual) == 0:
        raise ValueError("Calibration data is empty")
    if np.any(length <= 0) or np.any(breadth <= 0) or np.any(actual <= 0):
        raise ValueError("All axes and weights must be positive")

    feature = length * np.square(breadth)
    denominator = float(np.dot(feature, feature))
    if denominator <= 0:
        raise ValueError("Calibration feature has zero variance")

    k = float(np.dot(feature, actual) / denominator)
    predicted = k * feature
    residuals = actual - predicted
    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.dot(actual - actual.mean(), actual - actual.mean()))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

    sample_count = len(actual)
    residual_variance = ss_res / max(sample_count - 1, 1)
    standard_error = math.sqrt(residual_variance / denominator)
    within_3g = float(np.mean(np.abs(residuals) <= 3.0))

    return {
        "k": k,
        "formula": "weight_g = k * length_mm * breadth_mm^2",
        "samples": sample_count,
        "r2": r2,
        "mae_g": float(np.mean(np.abs(residuals))),
        "rmse_g": float(math.sqrt(np.mean(np.square(residuals)))),
        "mean_signed_error_g": float(np.mean(residuals)),
        "within_3g_rate": within_3g,
        "k_standard_error": standard_error,
        "k_95_ci": [
            k - 1.96 * standard_error,
            k + 1.96 * standard_error,
        ],
        "actual_weight_range_g": [
            float(actual.min()),
            float(actual.max()),
        ],
        "predicted_weight_range_g": [
            float(predicted.min()),
            float(predicted.max()),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/weight_calibration_report.json"),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum paired calibration records; use 30-50+ for release",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Missing calibration CSV: {args.data}")
    if args.min_samples < 1:
        raise ValueError("--min-samples must be positive")

    length, breadth, actual = load_records(args.data)
    if len(actual) < args.min_samples:
        raise ValueError(
            f"Only {len(actual)} calibration rows; need at least "
            f"{args.min_samples}"
        )

    report = fit_calibration(length, breadth, actual)
    report.update(
        {
            "data": str(args.data),
            "min_samples": args.min_samples,
            "release_ready": len(actual) >= 30,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Samples: {report['samples']}")
    print(f"k: {report['k']:.10f}")
    print(f"R2: {report['r2']:.6f}" if report["r2"] is not None else "R2: N/A")
    print(f"MAE: {report['mae_g']:.3f} g")
    print(f"RMSE: {report['rmse_g']:.3f} g")
    print(f"Within +/-3 g: {report['within_3g_rate']:.3%}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
