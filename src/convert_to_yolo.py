"""
Convert existing egg classification dataset to YOLO detection format.
Uses edge detection to find egg boundaries and generate bounding boxes.
"""
import os
import cv2
import numpy as np
import yaml
from pathlib import Path

DAMAGED_DIR = "data/damage/Damaged"
NOT_DAMAGED_DIR = "data/damage/Not Damaged"
OUTPUT_BASE = "data/detection"

def find_egg_contour(img):
    """Find egg contour using edge detection and return bounding box."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    h, w = img.shape[:2]
    img_area = w * h

    kernel = np.ones((5, 5), np.uint8)

    best_box = None
    best_area = 0

    # Try Otsu, then Canny, fallback to largest contour
    for method in ['otsu', 'canny']:
        if method == 'otsu':
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = cv2.Canny(blur, 30, 100)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            thresh = cv2.erode(thresh, kernel, iterations=1)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        method_best = None
        method_best_area = 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < 0.02 * img_area:
                continue
            if area > method_best_area:
                method_best_area = area
                method_best = (x, y, cw, ch)

        if method_best is not None:
            x, y, cw, ch = method_best
            tightness = cw * ch / img_area
            if tightness > 0.95:
                pad_x = int(cw * 0.02)
                pad_y = int(ch * 0.02)
            else:
                pad_x = int(cw * 0.15)
                pad_y = int(ch * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)

            if tightness > 0.98:
                x1, y1 = int(w * 0.05), int(h * 0.05)
                x2, y2 = int(w * 0.95), int(h * 0.95)

            box_area = (x2 - x1) * (y2 - y1)
            if box_area > best_area:
                best_area = box_area
                best_box = (x1, y1, x2, y2)

        if best_box is not None and method_best_area < 0.95 * img_area:
            return best_box

    if best_box is None:
        margin_x, margin_y = int(w * 0.1), int(h * 0.1)
        best_box = (margin_x, margin_y, w - margin_x, h - margin_y)

    return best_box

def convert_to_yolo(img_path, output_label, img_w, img_h, class_id=0):
    """Convert bbox to YOLO format."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    bbox = find_egg_contour(img)
    if bbox is None:
        x1, y1, x2, y2 = int(img_w * 0.1), int(img_h * 0.1), int(img_w * 0.9), int(img_h * 0.9)
    else:
        x1, y1, x2, y2 = bbox
    
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    box_w = (x2 - x1) / img_w
    box_h = (y2 - y1) / img_h
    
    with open(output_label, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
    return True

def main():
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{OUTPUT_BASE}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_BASE}/labels/{split}", exist_ok=True)
    
    # Detection has one class: egg. Damage is classified downstream.
    damaged = [(p, 1) for p in Path(DAMAGED_DIR).glob("*.jpg")]
    not_damaged = [(p, 0) for p in Path(NOT_DAMAGED_DIR).glob("*.jpg")]
    all_images = damaged + not_damaged
    np.random.shuffle(all_images)
    
    print(f"Total images: {len(all_images)}")
    print(f"  Not Damaged: {len(not_damaged)}")
    print(f"  Damaged: {len(damaged)}")
    
    # Stratified 70-10-20 split
    damaged_only = [x for x in all_images if x[1] == 1]
    not_damaged_only = [x for x in all_images if x[1] == 0]
    
    def split_list(lst, train_pct, val_pct):
        n = len(lst)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        train = lst[:n_train]
        val = lst[n_train:n_train + n_val]
        test = lst[n_train + n_val:]
        return train, val, test
    
    d_train, d_val, d_test = split_list(damaged_only, 0.7, 0.1)
    nd_train, nd_val, nd_test = split_list(not_damaged_only, 0.7, 0.1)
    
    splits = {
        'train': d_train + nd_train,
        'val': d_val + nd_val,
        'test': d_test + nd_test,
    }
    
    # Clean up stale cache files
    labels_dir = Path(OUTPUT_BASE) / "labels"
    if labels_dir.is_dir():
        for cache_file in labels_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception:
                pass

    for split_name, items in splits.items():
        np.random.shuffle(items)
        print(f"Processing {split_name} ({len(items)} images)...")
        for i, (img_path, original_cls) in enumerate(items):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            dest_img = f"{OUTPUT_BASE}/images/{split_name}/{img_path.name}"
            cv2.imwrite(dest_img, img)
            
            dest_label = f"{OUTPUT_BASE}/labels/{split_name}/{img_path.stem}.txt"
            convert_to_yolo(img_path, dest_label, w, h, class_id=0)
            
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(items)}")
    
    # Create data.yaml with 2 classes
    data_config = {
        'path': os.path.abspath(OUTPUT_BASE),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': {0: 'egg'}
    }
    with open(f"{OUTPUT_BASE}/data.yaml", 'w') as f:
        yaml.dump(data_config, f)
    
    print("\nDone! Dataset created:")
    total = sum(len(v) for v in splits.values())
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    print(f"  Total: {total}")
    print("\nNext: python src/train_yolo.py --train")

if __name__ == "__main__":
    main()