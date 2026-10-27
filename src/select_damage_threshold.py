"""Select a damaged-egg decision threshold on a validation manifest."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from dataset import EggDataset
from evaluate_classifier import choose_device, read_labels
from model_factory import build_model
from utils import get_transforms


def collect_probabilities(args, checkpoint):
    labels = read_labels(args.labels)
    device = choose_device()
    arch = str(checkpoint.get("arch", "custom"))
    arch_args = checkpoint.get("arch_args") or {}
    num_classes = int(arch_args.get("num_classes", 2))
    img_size = int(args.img_size or checkpoint.get("img_size", 224))
    dataset = EggDataset(
        args.image_root, labels, get_transforms(False, img_size)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = build_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    all_probabilities = []
    labels_seen = []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            all_probabilities.append(
                torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
            )
            labels_seen.extend(batch_labels.tolist())
    return np.concatenate(all_probabilities), np.asarray(labels_seen), device


def sweep(probabilities, labels, step):
    rows = []
    for threshold in np.arange(step, 1.0, step):
        predictions = np.where(probabilities >= threshold, 0, 1)
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "accuracy": float(accuracy_score(labels, predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, predictions)
                ),
                "damaged_recall": float(
                    recall_score(labels, predictions, pos_label=0)
                ),
                "damaged_precision": float(
                    precision_score(
                        labels, predictions, pos_label=0, zero_division=0
                    )
                ),
                "not_damaged_recall": float(
                    recall_score(labels, predictions, pos_label=1)
                ),
            }
        )
    return rows


def select(rows, min_damaged_recall, min_balanced_accuracy, min_not_damaged_recall):
    candidates = [
        row
        for row in rows
        if row["damaged_recall"] >= min_damaged_recall
        and row["balanced_accuracy"] >= min_balanced_accuracy
        and row["not_damaged_recall"] >= min_not_damaged_recall
    ]
    if candidates:
        return min(
            candidates,
            key=lambda row: (
                row["threshold"],
                -row["balanced_accuracy"],
                -row["damaged_recall"],
            ),
        )
    return max(
        rows,
        key=lambda row: (row["balanced_accuracy"], row["damaged_recall"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--min-damaged-recall", type=float, default=0.98)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.93)
    parser.add_argument("--min-not-damaged-recall", type=float, default=0.98)
    parser.add_argument("--update-checkpoint", action="store_true")
    args = parser.parse_args()

    if not 0 < args.step < 1:
        raise ValueError("--step must be between 0 and 1")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    prob_values, labels, device = collect_probabilities(args, checkpoint)
    rows = sweep(prob_values, labels, args.step)
    best = select(
        rows,
        args.min_damaged_recall,
        args.min_balanced_accuracy,
        args.min_not_damaged_recall,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "labels": str(args.labels),
        "device": str(device),
        "samples": int(len(labels)),
        "selection_criteria": {
            "min_damaged_recall": args.min_damaged_recall,
            "min_balanced_accuracy": args.min_balanced_accuracy,
            "min_not_damaged_recall": args.min_not_damaged_recall,
            "tie_break": "lowest threshold meeting the recall/balanced constraints",
        },
        "selected_threshold": best["threshold"],
        "selected_metrics": best,
        "candidates": [
            row
            for row in rows
            if row["damaged_recall"] >= args.min_damaged_recall
            and row["balanced_accuracy"] >= args.min_balanced_accuracy
            and row["not_damaged_recall"] >= args.min_not_damaged_recall
        ],
        "sweep": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    if args.update_checkpoint:
        checkpoint["damaged_threshold"] = best["threshold"]
        checkpoint["threshold_selection"] = {
            "labels": str(args.labels),
            "criteria": report["selection_criteria"],
            "selected_metrics": best,
        }
        temporary = Path(str(args.checkpoint) + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, args.checkpoint)
        print(f"Updated checkpoint threshold: {best['threshold']}")

    print(f"Samples: {len(labels)}")
    print(f"Selected threshold: {best['threshold']}")
    print(f"Validation metrics: {best}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
