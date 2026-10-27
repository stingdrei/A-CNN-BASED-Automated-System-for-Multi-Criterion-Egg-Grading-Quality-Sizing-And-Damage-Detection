"""Dump per-sample damage classifier outcomes to CSV for manual error review.

Label convention (preserved from the training pipeline):
    0 = Damaged, 1 = Not Damaged

`error_type` is derived from (true_label, pred_label):
    TP = damaged predicted damaged
    TN = not-damaged predicted not-damaged
    FN = damaged predicted not-damaged
    FP = not-damaged predicted damaged

The tool supports both the legacy custom CNN and ResNet18 checkpoints. It reads
the architecture, class order, image size, and damaged threshold from checkpoint
metadata, with CLI overrides for image size and threshold.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import EggDataset
from model_factory import ARCH_CUSTOM, build_model
from utils import get_transforms

EXPECTED_CLASSES = ["Damaged", "Not Damaged"]
ERROR_TYPES = ("TP", "TN", "FN", "FP")
CSV_COLUMNS = [
    "filename",
    "true_label",
    "pred_label",
    "damaged_probability",
    "confidence",
    "threshold",
    "error_type",
]


def read_labels(path: str):
    frame = pd.read_csv(path)
    return [(str(row.filename), int(row.label)) for row in frame.itertuples()]


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def error_type(true_label: int, pred_label: int) -> str:
    if true_label == pred_label:
        return "TP" if true_label == 0 else "TN"
    return "FP" if pred_label == 0 else "FN"


def resolve_checkpoint_config(
    checkpoint: dict,
    img_size: int | None = None,
    threshold: float | None = None,
) -> dict:
    """Resolve architecture and decision metadata from a classifier checkpoint."""
    class_names = [str(name) for name in checkpoint.get("class_names", EXPECTED_CLASSES)]
    if class_names != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected checkpoint class names: {class_names!r}")
    arch = str(checkpoint.get("arch", ARCH_CUSTOM))
    arch_args = checkpoint.get("arch_args") or {}
    num_classes = int(arch_args.get("num_classes", len(class_names)))
    if num_classes != len(class_names):
        raise ValueError(
            f"Checkpoint num_classes={num_classes} does not match "
            f"class_names={class_names!r}"
        )
    resolved_img_size = int(img_size or checkpoint.get("img_size", 224))
    if resolved_img_size <= 0:
        raise ValueError(f"Invalid image size: {resolved_img_size}")
    if threshold is None:
        threshold = float(checkpoint.get("damaged_threshold", 0.5))
        threshold_source = "checkpoint"
    else:
        threshold = float(threshold)
        threshold_source = "cli"
    if not 0 < threshold < 1:
        raise ValueError(f"Invalid damaged threshold: {threshold}")
    return {
        "arch": arch,
        "arch_args": arch_args,
        "class_names": class_names,
        "num_classes": num_classes,
        "img_size": resolved_img_size,
        "damaged_threshold": threshold,
        "damaged_threshold_source": threshold_source,
    }


def classify_probability(damaged_probability: float, threshold: float) -> tuple[int, float]:
    """Return (label, predicted-class confidence) for a damaged probability."""
    if not 0 <= damaged_probability <= 1:
        raise ValueError(f"Invalid damaged probability: {damaged_probability}")
    if damaged_probability >= threshold:
        return 0, damaged_probability
    return 1, 1.0 - damaged_probability


def collect_errors(
    labels,
    image_root: str,
    checkpoint: str,
    batch_size: int = 32,
    img_size: int | None = None,
    threshold: float | None = None,
    device: torch.device = None,
):
    device = device or choose_device()
    checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
    config = resolve_checkpoint_config(checkpoint_obj, img_size, threshold)
    state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)

    dataset = EggDataset(
        image_root,
        labels,
        get_transforms(False, config["img_size"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = build_model(
        config["arch"],
        num_classes=config["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rows = []
    index = 0
    with torch.no_grad():
        for images, _ in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            damaged_probabilities = probabilities[:, 0].cpu().tolist()
            for damaged_probability in damaged_probabilities:
                filename, true_label = labels[index]
                pred_label, confidence = classify_probability(
                    float(damaged_probability), config["damaged_threshold"]
                )
                rows.append(
                    {
                        "filename": filename,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "damaged_probability": round(float(damaged_probability), 6),
                        "confidence": round(float(confidence), 6),
                        "threshold": config["damaged_threshold"],
                        "error_type": error_type(true_label, pred_label),
                    }
                )
                index += 1
    return rows, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    labels = read_labels(args.labels)
    rows, config = collect_errors(
        labels,
        args.image_root,
        args.checkpoint,
        batch_size=args.batch_size,
        img_size=args.img_size,
        threshold=args.threshold,
    )
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    counts = frame["error_type"].value_counts().to_dict()
    print(
        f"Arch: {config['arch']}  Threshold: "
        f"{config['damaged_threshold']:.4f} "
        f"({config['damaged_threshold_source']})"
    )
    print(f"Samples: {len(frame)}")
    print(f"FN: {counts.get('FN', 0)}  FP: {counts.get('FP', 0)}")
    print(f"Saved audit: {args.output}")


if __name__ == "__main__":
    main()
