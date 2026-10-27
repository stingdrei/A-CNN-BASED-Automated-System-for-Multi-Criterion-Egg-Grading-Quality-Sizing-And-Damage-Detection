"""Evaluate a binary damage checkpoint on an untouched manifest."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from dataset import EggDataset
from model_factory import ARCH_CUSTOM, build_model
from utils import get_transforms


def read_labels(path: str):
    frame = pd.read_csv(path)
    return [(str(row.filename), int(row.label)) for row in frame.itertuples()]


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(args: argparse.Namespace) -> dict:
    labels = read_labels(args.labels)
    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    class_names = checkpoint.get("class_names", ["Damaged", "Not Damaged"])
    if list(class_names) != ["Damaged", "Not Damaged"]:
        raise ValueError(f"Unexpected checkpoint class names: {class_names!r}")
    arch = str(checkpoint.get("arch", ARCH_CUSTOM))
    arch_args = checkpoint.get("arch_args") or {}
    num_classes = int(arch_args.get("num_classes", len(class_names)))
    if num_classes != len(class_names):
        raise ValueError(
            f"Checkpoint num_classes={num_classes} does not match "
            f"class_names={class_names!r}"
        )
    img_size = int(args.img_size or checkpoint.get("img_size", 224))
    if img_size <= 0:
        raise ValueError(f"Invalid image size: {img_size}")
    if args.threshold is None:
        threshold = float(checkpoint.get("damaged_threshold", 0.5))
        threshold_source = "checkpoint"
    else:
        threshold = float(args.threshold)
        threshold_source = "cli"
    if not 0 < threshold < 1:
        raise ValueError(f"Invalid damaged threshold: {threshold}")
    dataset = EggDataset(
        args.image_root, labels, get_transforms(False, img_size)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = build_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            damaged_probability = probabilities[:, 0].cpu().tolist()
            predictions = [
                0 if probability >= threshold else 1
                for probability in damaged_probability
            ]
            y_true.extend(batch_labels.tolist())
            y_pred.extend(predictions)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    damaged_total = int(matrix[0].sum())
    damaged_recall = float(matrix[0, 0] / damaged_total) if damaged_total else 0.0
    result = {
        "checkpoint": str(args.checkpoint),
        "arch": arch,
        "img_size": img_size,
        "damaged_threshold": threshold,
        "damaged_threshold_source": threshold_source,
        "samples": len(y_true),
        "device": str(device),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "damaged_recall": damaged_recall,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Damaged", "Not Damaged"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Damaged probability threshold; defaults to checkpoint metadata "
            "or 0.5"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args)
    print(f"Device: {result['device']}")
    print(f"Samples: {result['samples']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Balanced accuracy: {result['balanced_accuracy']:.4f}")
    print(f"Macro precision: {result['macro_precision']:.4f}")
    print(f"Macro recall: {result['macro_recall']:.4f}")
    print(f"Macro F1: {result['macro_f1']:.4f}")
    print(f"Damaged threshold: {result['damaged_threshold']:.4f}")
    print(f"Damaged recall: {result['damaged_recall']:.4f}")
    print("Confusion matrix [Damaged, Not Damaged]:")
    for row in result["confusion_matrix"]:
        print(row)
    print("\nClassification report:")
    print(json.dumps(result["classification_report"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
