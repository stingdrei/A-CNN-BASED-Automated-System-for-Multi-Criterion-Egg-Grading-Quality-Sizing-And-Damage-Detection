"""Create stratified damage-classification manifests from class folders.

Example:
    python src/create_damage_split.py \
      --source "data/damage" \
      --output data/damage/manifests
"""

import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_LABELS = {"Damaged": 0, "Not Damaged": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(source: Path) -> pd.DataFrame:
    rows = []
    for class_name, label in CLASS_LABELS.items():
        class_dir = source / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"filename": str(path), "label": label})
    if not rows:
        raise ValueError(f"No supported images found below {source}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/damage"))
    parser.add_argument("--output", type=Path, default=Path("data/damage/manifests"))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.val_size <= 0 or args.test_size <= 0 or args.val_size + args.test_size >= 1:
        raise ValueError("val-size and test-size must be positive and sum to less than 1")

    data = collect(args.source)
    train, remainder = train_test_split(
        data,
        test_size=args.val_size + args.test_size,
        random_state=args.seed,
        stratify=data["label"],
    )
    relative_test = args.test_size / (args.val_size + args.test_size)
    val, test = train_test_split(
        remainder,
        test_size=relative_test,
        random_state=args.seed,
        stratify=remainder["label"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.sample(frac=1, random_state=args.seed).to_csv(
            args.output / f"{name}.csv", index=False
        )
        print(f"{name}: {len(frame)} images")
    print(f"total: {len(data)} images")


if __name__ == "__main__":
    main()
