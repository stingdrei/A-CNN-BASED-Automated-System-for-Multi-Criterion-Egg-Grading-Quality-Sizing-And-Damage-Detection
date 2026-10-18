"""Dump per-sample damage classifier outcomes to CSV for manual error review.

Label convention (preserved from the training pipeline):
    0 = Damaged, 1 = Not Damaged

`error_type` is derived from (true_label, pred_label):
    TP = damaged predicted damaged
    TN = not-damaged predicted not-damaged
    FN = damaged predicted not-damaged
    FP = not-damaged predicted damaged

`confidence` is the softmax probability of the predicted class.
This tool only reports; it never edits manifests or labels.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import EggDataset
from model import EggGradingCNN
from utils import get_transforms

ERROR_TYPES = ("TP", "TN", "FN", "FP")
CSV_COLUMNS = ["filename", "true_label", "pred_label", "confidence", "error_type"]


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


def collect_errors(
    labels,
    image_root: str,
    checkpoint: str,
    batch_size: int = 32,
    img_size: int = 224,
    device: torch.device = None,
):
    dataset = EggDataset(image_root, labels, get_transforms(False, img_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = device or choose_device()

    checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
    model = EggGradingCNN(num_classes=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rows = []
    index = 0
    with torch.no_grad():
        for images, _ in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            confidences, predictions = probabilities.max(dim=1)
            for pred, confidence in zip(
                predictions.cpu().tolist(), confidences.cpu().tolist()
            ):
                filename, true_label = labels[index]
                rows.append(
                    {
                        "filename": filename,
                        "true_label": true_label,
                        "pred_label": int(pred),
                        "confidence": round(float(confidence), 6),
                        "error_type": error_type(true_label, int(pred)),
                    }
                )
                index += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    labels = read_labels(args.labels)
    rows = collect_errors(
        labels,
        args.image_root,
        args.checkpoint,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    counts = frame["error_type"].value_counts().to_dict()
    print(f"Samples: {len(frame)}")
    print(f"FN: {counts.get('FN', 0)}  FP: {counts.get('FP', 0)}")
    print(f"Saved audit: {args.output}")


if __name__ == "__main__":
    main()
