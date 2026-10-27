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

WEIGHT_DECAY = 1e-4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def parse_class_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if len(names) != 2:
        raise ValueError("--class-names must contain exactly two comma-separated names")
    return names


def metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: list[str],
) -> Dict[str, object]:
    true, pred = list(y_true), list(y_pred)
    per_class_precision = precision_score(
        true, pred, labels=[0, 1], average=None, zero_division=0
    )
    per_class_recall = recall_score(
        true, pred, labels=[0, 1], average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        true, pred, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "accuracy": accuracy_score(true, pred),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
        "precision": precision_score(true, pred, zero_division=0),
        "recall": recall_score(true, pred, zero_division=0),
        "f1": f1_score(true, pred, zero_division=0),
        "class_0_precision": float(per_class_precision[0]),
        "class_0_recall": float(per_class_recall[0]),
        "class_0_f1": float(per_class_f1[0]),
        "class_1_precision": float(per_class_precision[1]),
        "class_1_recall": float(per_class_recall[1]),
        "class_1_f1": float(per_class_f1[1]),
        "report": classification_report(
            true,
            pred,
            labels=[0, 1],
            target_names=class_names,
            zero_division=0,
        ),
    }


def evaluate(
    model, loader, device, class_names: list[str]
) -> Tuple[float, Dict[str, object]]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            total_loss += loss_fn(logits, labels.to(device)).item() * len(labels)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    return total_loss / max(len(loader.dataset), 1), metrics(
        y_true, y_pred, class_names
    )


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
    class_names: list[str],
) -> dict:
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "img_size": img_size,
        "seed": seed,
        "best_val_metrics": record,
    }
    checkpoint.update(
        checkpoint_metadata(
            arch, num_classes=len(class_names), pretrained=pretrained
        )
    )
    checkpoint["hyperparams"] = dict(hyperparams)
    return checkpoint


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else choose_device()
    print(f"Device: {device}")
    class_names = parse_class_names(args.class_names)
    train_labels, val_labels = read_labels(args.labels), read_labels(args.val_labels)
    validate_paths(args.image_root, train_labels + val_labels)
    invalid_labels = sorted(
        {label for _, label in train_labels + val_labels}
        - set(range(len(class_names)))
    )
    if invalid_labels:
        raise ValueError(
            f"Labels {invalid_labels} are outside class_names={class_names!r}"
        )
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
    stage1_lr = args.stage1_lr if args.stage1_lr is not None else args.lr
    stage2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr
    hyperparams = {
        "lr": args.lr,
        "stage1_lr": stage1_lr,
        "stage2_lr": stage2_lr,
        "batch_size": args.batch_size,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "total_epochs": stage1_epochs + stage2_epochs,
        "img_size": args.img_size,
        "seed": args.seed,
        "weight_decay": WEIGHT_DECAY,
        "init_checkpoint": args.init_checkpoint,
        "class_names": class_names,
        "select_metric": args.select_metric,
    }

    model = build_model(
        args.arch, num_classes=len(class_names), pretrained=args.pretrained
    )
    if args.init_checkpoint:
        init = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        init_arch = init.get("arch") or ARCH_CUSTOM
        if init_arch != args.arch:
            raise ValueError(
                f"Initial checkpoint arch={init_arch!r} does not match "
                f"--arch={args.arch!r}"
            )
        model.load_state_dict(init.get("state_dict", init))
        print(f"Initialized from {args.init_checkpoint} ({init_arch})")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    best_score, epoch_number = -1.0, 0
    history = []
    optimizer = None

    def run_epochs(epochs: int, stage_name: str) -> None:
        nonlocal best_score, epoch_number
        for _ in range(epochs):
            epoch_number += 1
            model.train()
            if stage_name == "stage1" and args.arch == ARCH_RESNET18:
                set_frozen_bn_eval(model)
            for images, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss_fn(model(images.to(device)), labels.to(device)).backward()
                optimizer.step()
            val_loss, val_metrics = evaluate(
                model, val_loader, device, class_names
            )
            record = {
                "epoch": epoch_number,
                "stage": stage_name,
                "val_loss": val_loss,
                **val_metrics,
            }
            record.pop("report", None)
            history.append(record)
            score = float(val_metrics[args.select_metric])
            if score > best_score:
                best_score = score
                checkpoint = build_checkpoint(
                    model,
                    args.arch,
                    args.pretrained,
                    args.img_size,
                    args.seed,
                    record,
                    hyperparams,
                    class_names,
                )
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, args.output)

    if stage1_epochs > 0:
        set_backbone_frozen(model, args.arch)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=stage1_lr,
            weight_decay=WEIGHT_DECAY,
        )
        run_epochs(stage1_epochs, "stage1")

    if stage2_epochs > 0:
        unfreeze_all(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=stage2_lr, weight_decay=WEIGHT_DECAY
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
    parser.add_argument("--stage1-lr", type=float, default=None)
    parser.add_argument("--stage2-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, or cuda:0")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--class-names",
        default="Damaged,Not Damaged",
        help="Two comma-separated class names in label-index order",
    )
    parser.add_argument(
        "--select-metric",
        choices=["balanced_accuracy", "macro_f1", "recall", "f1"],
        default="balanced_accuracy",
        help="Validation metric used to save the best checkpoint",
    )
    parser.add_argument(
        "--arch", choices=[ARCH_CUSTOM, ARCH_RESNET18], default=ARCH_CUSTOM
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=0)
    parser.add_argument("--stage2-epochs", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
