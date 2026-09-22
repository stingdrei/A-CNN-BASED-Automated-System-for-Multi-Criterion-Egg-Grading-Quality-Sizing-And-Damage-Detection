"""Reproducible binary classifier training and evaluation.

Usage:
    python src/train_classifier.py --labels data/damage/manifests/train.csv \
      --val-labels data/damage/manifests/val.csv --image-root . \
      --output models/egg_grader.pth
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import EggDataset
from model import EggGradingCNN
from utils import get_transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_labels(path: str) -> list[Tuple[str, int]]:
    frame = pd.read_csv(path)
    required = {"filename", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain filename and label columns")
    return [(str(row.filename), int(row.label)) for row in frame.itertuples()]


def validate_paths(image_root: str, labels: list[Tuple[str, int]]) -> None:
    missing = [
        str(Path(image_root) / filename)
        for filename, _ in labels
        if not (Path(image_root) / filename).is_file()
    ]
    if missing:
        examples = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} training images are missing below {image_root}; "
            f"examples: {examples}"
        )


def metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, object]:
    true, pred = list(y_true), list(y_pred)
    return {
        "accuracy": accuracy_score(true, pred),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "precision": precision_score(true, pred, zero_division=0),
        "recall": recall_score(true, pred, zero_division=0),
        "f1": f1_score(true, pred, zero_division=0),
        "report": classification_report(true, pred, zero_division=0),
    }


def evaluate(model, loader, device) -> Tuple[float, Dict[str, object]]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            total_loss += loss_fn(logits, labels.to(device)).item() * len(labels)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    return total_loss / max(len(loader.dataset), 1), metrics(y_true, y_pred)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_labels, val_labels = read_labels(args.labels), read_labels(args.val_labels)
    validate_paths(args.image_root, train_labels + val_labels)
    train_ds = EggDataset(args.image_root, train_labels, get_transforms(True, args.img_size))
    val_ds = EggDataset(
        args.image_root, val_labels, get_transforms(False, args.img_size)
    )
    counts = np.bincount([label for _, label in train_labels], minlength=2)
    weights = [1.0 / max(counts[label], 1) for _, label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False)
    model = EggGradingCNN(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(images.to(device)), labels.to(device)).backward()
            optimizer.step()
        val_loss, val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch + 1, "val_loss": val_loss, **val_metrics}
        record.pop("report", None)
        history.append(record)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": ["Damaged", "Not Damaged"],
                    "img_size": args.img_size,
                    "seed": args.seed,
                    "best_val_metrics": record,
                },
                args.output,
            )
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(history, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="models/egg_grader.pth")
    parser.add_argument("--metrics", default="models/damage_metrics.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
