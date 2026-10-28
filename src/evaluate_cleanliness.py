"""Evaluate a Clean/Stained checkpoint and select a stain-recall threshold.

The threshold is selected on the validation manifest only. The untouched test
manifest is then evaluated at that fixed threshold. ``P(Stained)`` is class 1.

Example:
    python src/evaluate_cleanliness.py \
      --valid-labels data/cleanliness_candidates/manifests/valid.csv \
      --test-labels data/cleanliness_candidates/manifests/test.csv \
      --checkpoint models/egg_grader_cleanliness_v1.pth \
      --image-root . \
      --target-stained-recall 0.98 \
      --output reports/cleanliness_crop_resnet18_v1_eval.json \
      --errors-output reports/cleanliness_crop_resnet18_v1_eval_errors.csv
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from dataset import EggDataset
from model_factory import build_model
from train_classifier import choose_device, read_labels, validate_paths
from utils import get_transforms

CLEAN, STAINED = 0, 1
CLASS_NAMES = ["Clean", "Stained"]


def validate_checkpoint_classes(class_names: Iterable[str]) -> list[str]:
    names = [str(name) for name in class_names]
    if names != CLASS_NAMES:
        raise ValueError(
            f"Expected cleanliness class names {CLASS_NAMES!r}, got {names!r}"
        )
    return names


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cleanliness_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    class_names = validate_checkpoint_classes(
        checkpoint.get("class_names", [])
    )
    arch = str(checkpoint.get("arch", "custom"))
    arch_args = checkpoint.get("arch_args") or {}
    num_classes = int(arch_args.get("num_classes", len(class_names)))
    if num_classes != len(class_names):
        raise ValueError(
            f"Checkpoint num_classes={num_classes} does not match "
            f"class_names={class_names!r}"
        )
    img_size = int(checkpoint.get("img_size", 224))
    if img_size <= 0:
        raise ValueError(f"Invalid checkpoint img_size: {img_size}")

    model = build_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, {
        "arch": arch,
        "class_names": class_names,
        "img_size": img_size,
        "checkpoint_metadata": {
            key: checkpoint[key]
            for key in ("stained_threshold", "best_val_metrics")
            if key in checkpoint
        },
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
    }


def predict_stained_probabilities(
    model,
    labels: list[tuple[str, int]],
    image_root: str,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = EggDataset(
        image_root, labels, get_transforms(False, img_size)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities: list[float] = []
    with torch.no_grad():
        for images, _ in loader:
            batch = torch.softmax(model(images.to(device)), dim=1)
            probabilities.extend(batch[:, STAINED].cpu().tolist())
    return np.asarray(probabilities, dtype=float)


def expected_calibration_error(
    y_true: np.ndarray,
    stained_probabilities: np.ndarray,
    threshold: float,
    bins: int = 10,
) -> float:
    predictions = (stained_probabilities >= threshold).astype(int)
    confidence = np.where(
        predictions == STAINED,
        stained_probabilities,
        1.0 - stained_probabilities,
    )
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidence[mask].mean())
        error += float(mask.sum()) / total * abs(bin_accuracy - bin_confidence)
    return error


def threshold_metrics(
    y_true: np.ndarray,
    stained_probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (stained_probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[CLEAN, STAINED])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    clean_total, stained_total = tn + fp, tp + fn
    return {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "stained_precision": float(
            precision_score(
                y_true, predictions, pos_label=STAINED, zero_division=0
            )
        ),
        "stained_recall": float(
            recall_score(
                y_true, predictions, pos_label=STAINED, zero_division=0
            )
        ),
        "stained_f1": float(
            f1_score(
                y_true, predictions, pos_label=STAINED, zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "clean_recall": float(tn / clean_total) if clean_total else 0.0,
        "false_positive_rate": float(fp / clean_total)
        if clean_total
        else 0.0,
        "false_negative_rate": float(fn / stained_total)
        if stained_total
        else 0.0,
        "expected_calibration_error": expected_calibration_error(
            y_true, stained_probabilities, threshold
        ),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    stained_probabilities: np.ndarray,
    step: float = 0.01,
) -> list[dict]:
    if not 0 < step < 1:
        raise ValueError(f"Invalid threshold step: {step}")
    thresholds = np.round(np.arange(step, 1.0, step), 10)
    return [
        threshold_metrics(y_true, stained_probabilities, float(threshold))
        for threshold in thresholds
    ]


def select_threshold(
    validation_metrics: list[dict],
    target_stained_recall: float,
) -> dict:
    if not 0 <= target_stained_recall <= 1:
        raise ValueError(
            f"Invalid target stained recall: {target_stained_recall}"
        )
    eligible = [
        row
        for row in validation_metrics
        if row["stained_recall"] >= target_stained_recall
    ]
    selection_note = (
        f"stained_recall >= {target_stained_recall:.2f}"
        if eligible
        else "no threshold met the target; maximized stained recall"
    )
    pool = eligible or validation_metrics
    # Prefer balanced accuracy, then recall, then the more conservative
    # (lower) stain threshold. Test data is never used for this decision.
    selected = sorted(
        pool,
        key=lambda row: (
            -row["balanced_accuracy"],
            -row["stained_recall"],
            row["threshold"],
        ),
    )[0]
    return {
        "target_stained_recall": float(target_stained_recall),
        "criterion": "max balanced_accuracy subject to recall floor",
        "eligible_thresholds": len(eligible),
        "note": selection_note,
        "selected_threshold": selected["threshold"],
        "selected_metrics": selected,
    }


def evaluate_split(
    labels: list[tuple[str, int]],
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray([label for _, label in labels], dtype=int)
    return threshold_metrics(y_true, probabilities, threshold)


def error_rows(
    split: str,
    labels: list[tuple[str, int]],
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict]:
    rows = []
    for (filename, true_label), probability in zip(labels, probabilities):
        prediction = STAINED if probability >= threshold else CLEAN
        if prediction == true_label:
            continue
        rows.append(
            {
                "split": split,
                "filename": filename,
                "true_label": CLASS_NAMES[true_label],
                "predicted_label": CLASS_NAMES[prediction],
                "stained_probability": f"{float(probability):.8f}",
                "outcome": "false_positive"
                if true_label == CLEAN
                else "false_negative",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-labels", required=True)
    parser.add_argument("--test-labels")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--target-stained-recall", type=float, default=0.98)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--errors-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else choose_device()
    model, metadata = load_cleanliness_model(args.checkpoint, device)
    img_size = int(metadata["img_size"])

    valid_labels = read_labels(args.valid_labels)
    validate_paths(args.image_root, valid_labels)
    valid_probabilities = predict_stained_probabilities(
        model,
        valid_labels,
        args.image_root,
        img_size,
        args.batch_size,
        device,
    )
    valid_true = np.asarray([label for _, label in valid_labels], dtype=int)
    sweep = sweep_thresholds(
        valid_true, valid_probabilities, args.threshold_step
    )
    selection = select_threshold(sweep, args.target_stained_recall)
    selected_threshold = selection["selected_threshold"]
    valid_metrics = evaluate_split(
        valid_labels, valid_probabilities, selected_threshold
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "img_size": img_size,
        "class_names": metadata["class_names"],
        "arch": metadata["arch"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_metadata": metadata["checkpoint_metadata"],
        "threshold_policy": {
            "selection_split": "valid",
            "target_stained_recall": float(args.target_stained_recall),
            "threshold_step": float(args.threshold_step),
            **selection,
        },
        "valid": valid_metrics,
        "valid_at_0_5": evaluate_split(valid_labels, valid_probabilities, 0.5),
        "threshold_sweep": sweep,
    }

    all_errors = error_rows(
        "valid", valid_labels, valid_probabilities, selected_threshold
    )
    if args.test_labels:
        test_labels = read_labels(args.test_labels)
        validate_paths(args.image_root, test_labels)
        test_probabilities = predict_stained_probabilities(
            model,
            test_labels,
            args.image_root,
            img_size,
            args.batch_size,
            device,
        )
        report["test"] = evaluate_split(
            test_labels, test_probabilities, selected_threshold
        )
        report["test_at_0_5"] = evaluate_split(
            test_labels, test_probabilities, 0.5
        )
        all_errors.extend(
            error_rows(
                "test",
                test_labels,
                test_probabilities,
                selected_threshold,
            )
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if args.errors_output:
        args.errors_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            all_errors,
            columns=[
                "split",
                "filename",
                "true_label",
                "predicted_label",
                "stained_probability",
                "outcome",
            ],
        ).to_csv(args.errors_output, index=False)

    print(f"Device: {device}")
    print(f"Selected threshold: {selected_threshold:.4f}")
    print(
        "Valid: "
        f"balanced_accuracy={valid_metrics['balanced_accuracy']:.4f} "
        f"stained_recall={valid_metrics['stained_recall']:.4f} "
        f"stained_precision={valid_metrics['stained_precision']:.4f} "
        f"ECE={valid_metrics['expected_calibration_error']:.4f}"
    )
    if "test" in report:
        test_metrics = report["test"]
        print(
            "Test: "
            f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} "
            f"stained_recall={test_metrics['stained_recall']:.4f} "
            f"stained_precision={test_metrics['stained_precision']:.4f} "
            f"ECE={test_metrics['expected_calibration_error']:.4f}"
        )
        print(
            "Test confusion matrix [[TN, FP], [FN, TP]]: "
            f"{test_metrics['confusion_matrix']}"
        )
    if args.output:
        print(f"Saved report: {args.output}")
    if args.errors_output:
        print(f"Saved errors: {args.errors_output}")


if __name__ == "__main__":
    main()
