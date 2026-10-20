"""Reproducible binary classifier training and evaluation.

Usage:
    python src/train_classifier.py --labels data/damage/manifests/train.csv \
      --val-labels data/damage/manifests/val.csv --image-root . \
      --output models/egg_grader.pth

Two-stage transfer-learning example:
    python src/train_classifier.py --labels ... --val-labels ... \
      --image-root . --arch resnet18 --pretrained \
      --stage1-epochs 3 --stage2-epochs 12
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
from model_factory import (
    ARCH_CUSTOM,
    ARCH_RESNET18,
    build_model,
    checkpoint_metadata,
)
from utils import get_transforms

CLASS_NAMES = ["Damaged", "Not Damaged"]
WEIGHT_DECAY = 1e-4


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


def resolve_stage_epochs(
    total_epochs: int, stage1_epochs: int, stage2_epochs: int
) -> Tuple[int, int]:
    """Return (stage1, stage2) epoch counts.

    Explicit stage counts win; otherwise fall back to a single full-network
    stage for the legacy ``--epochs`` value.
    """
    if stage1_epochs < 0 or stage2_epochs < 0:
        raise ValueError("stage epochs must be non-negative")
    if stage1_epochs + stage2_epochs > 0:
        return stage1_epochs, stage2_epochs
    return 0, total_epochs


def head_module(model, arch: str):
    if arch == ARCH_RESNET18:
        return model.fc
    if arch == ARCH_CUSTOM:
        return model.fc2
    raise ValueError(f"Unknown architecture {arch!r}")


def set_backbone_frozen(model, arch: str) -> None:
    """Freeze the backbone and keep only the classification head trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head_module(model, arch).parameters():
        parameter.requires_grad = True


def unfreeze_all(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def set_frozen_bn_eval(model) -> None:
    """Prevent frozen BatchNorm layers from updating running statistics."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def build_checkpoint(
    model,
    arch: str,
    pretrained: bool,
    img_size: int,
    seed: int,
    record: dict,
    hyperparams: dict,
) -> dict:
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": list(CLASS_NAMES),
        "img_size": img_size,
        "seed": seed,
        "best_val_metrics": record,
    }
    checkpoint.update(
        checkpoint_metadata(arch, num_classes=2, pretrained=pretrained)
    )
    checkpoint["hyperparams"] = dict(hyperparams)
    return checkpoint


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

    stage1_epochs, stage2_epochs = resolve_stage_epochs(
        args.epochs, args.stage1_epochs, args.stage2_epochs
    )
    hyperparams = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "total_epochs": stage1_epochs + stage2_epochs,
        "img_size": args.img_size,
        "seed": args.seed,
        "weight_decay": WEIGHT_DECAY,
    }

    model = build_model(
        args.arch, num_classes=2, pretrained=args.pretrained
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, epoch_number = -1.0, 0
    history = []
    optimizer = None

    def run_epochs(epochs: int, stage_name: str) -> None:
        nonlocal best_f1, epoch_number
        for _ in range(epochs):
            epoch_number += 1
            model.train()
            if stage_name == "stage1" and args.arch == ARCH_RESNET18:
                set_frozen_bn_eval(model)
            for images, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss_fn(model(images.to(device)), labels.to(device)).backward()
                optimizer.step()
            val_loss, val_metrics = evaluate(model, val_loader, device)
            record = {
                "epoch": epoch_number,
                "stage": stage_name,
                "val_loss": val_loss,
                **val_metrics,
            }
            record.pop("report", None)
            history.append(record)
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                checkpoint = build_checkpoint(
                    model,
                    args.arch,
                    args.pretrained,
                    args.img_size,
                    args.seed,
                    record,
                    hyperparams,
                )
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, args.output)

    if stage1_epochs > 0:
        set_backbone_frozen(model, args.arch)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
        )
        run_epochs(stage1_epochs, "stage1")

    if stage2_epochs > 0:
        unfreeze_all(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
        )
        run_epochs(stage2_epochs, "stage2")

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
    parser.add_argument(
        "--arch", choices=[ARCH_CUSTOM, ARCH_RESNET18], default=ARCH_CUSTOM
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=0)
    parser.add_argument("--stage2-epochs", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
