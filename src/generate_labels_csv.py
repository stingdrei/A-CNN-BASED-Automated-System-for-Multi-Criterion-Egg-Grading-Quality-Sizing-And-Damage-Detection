import os
import csv

# Source and output directories
SRC_ROOT = "data/damage"
OUT_CSV = "data/processed/train_labels.csv"

# Supported extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    class_folders = [
        name for name in os.listdir(SRC_ROOT)
        if os.path.isdir(os.path.join(SRC_ROOT, name))
    ]
    class_folders.sort()  # Alphabetical for reproducibility
    label_map = {folder: idx for idx, folder in enumerate(class_folders)}
    print("Label mapping:")
    for folder, idx in label_map.items():
        print(f"  {folder}: {idx}")

    rows = []
    for folder in class_folders:
        folder_path = os.path.join(SRC_ROOT, folder)
        for fname in os.listdir(folder_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                rows.append({
                    "filename": f"{folder}/{fname}",
                    "label": label_map[folder]
                })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} entries to {OUT_CSV}")

if __name__ == "__main__":
    main()
