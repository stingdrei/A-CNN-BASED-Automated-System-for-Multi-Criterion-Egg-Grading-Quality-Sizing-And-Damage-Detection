"""Fit W = k * L * B^2 from known-weight calibration measurements."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def fit(path: str) -> dict:
    data = pd.read_csv(path)
    required = {"length_mm", "breadth_mm", "actual_weight_g"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")
    data = data.dropna(subset=required)
    if len(data) < 10:
        raise ValueError("At least 10 known-weight eggs are required")
    x = data.length_mm.to_numpy() * data.breadth_mm.to_numpy() ** 2
    y = data.actual_weight_g.to_numpy()
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Calibration dimensions and weights must be positive")
    k = float(np.dot(x, y) / np.dot(x, x))
    predicted = k * x
    return {
        "k": k,
        "samples": int(len(data)),
        "r2": float(r2_score(y, predicted)),
        "mae_g": float(mean_absolute_error(y, predicted)),
        "rmse_g": float(mean_squared_error(y, predicted) ** 0.5),
        "within_3g": float(np.mean(np.abs(predicted - y) <= 3.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV with length_mm,breadth_mm,actual_weight_g")
    parser.add_argument("--output", default="models/weight_calibration.json")
    args = parser.parse_args()
    result = fit(args.csv)
    Path(args.output).write_text(pd.Series(result).to_json(indent=2))
    print(pd.Series(result).to_string())


if __name__ == "__main__":
    main()
