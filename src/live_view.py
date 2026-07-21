import cv2
import numpy as np
import time
from collections import defaultdict
import csv
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
from model import EggGradingCNN
from torchvision import transforms


class ObjectTracker:
    """Simple centroid tracking to prevent duplicate counts across frames."""
    
    def __init__(self, max_distance=50, max_disappeared=30):
        self.objects = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.disappeared = defaultdict(int)
    
    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return []
        
        new_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            new_centroids.append((cx, cy, det))
        
        used_detections = set()
        output = []
        
        if len(self.objects) == 0:
            for cent_idx, (cx, cy, det) in enumerate(new_centroids):
                self.objects[self.next_id] = (cx, cy, 0)
                self.disappeared[self.next_id] = 0
                output.append((self.next_id, det[0], det[1], det[2], det[3], det[4], True))
                self.next_id += 1
                used_detections.add(cent_idx)
        else:
            for obj_id in list(self.objects.keys()):
                last_cx, last_cy, _ = self.objects[obj_id]
                best_dist = float('inf')
                best_idx = -1
                
                for cent_idx, (cx, cy, det) in enumerate(new_centroids):
                    if cent_idx in used_detections:
                        continue
                    dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = cent_idx
                
                if best_idx != -1 and best_dist < self.max_distance:
                    cx, cy, det = new_centroids[best_idx]
                    self.objects[obj_id] = (cx, cy, self.objects[obj_id][2] + 1)
                    self.disappeared[obj_id] = 0
                    output.append((obj_id, det[0], det[1], det[2], det[3], det[4], False))
                    used_detections.add(best_idx)
                else:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        del self.objects[obj_id]
                        del self.disappeared[obj_id]
            
            for cent_idx, (cx, cy, det) in enumerate(new_centroids):
                if cent_idx not in used_detections:
                    self.objects[self.next_id] = (cx, cy, 0)
                    self.disappeared[self.next_id] = 0
                    output.append((self.next_id, det[0], det[1], det[2], det[3], det[4], True))
                    self.next_id += 1
        
        return output


class CameraCalibrator:
    def __init__(self):
        self.mm_per_pixel = 0.09
    
    def pixels_to_mm(self, pixels):
        return pixels * self.mm_per_pixel
    
    def estimate_weight(self, diameter_mm):
        if diameter_mm <= 0:
            return -1
        weight_g = 0.0005 * (diameter_mm ** 3)
        weight_g = max(30, min(80, weight_g))
        return round(weight_g, 1)
    
    def categorize_egg(self, diameter_px):
        if diameter_px < 0:
            return "N/A", -1
        diameter_mm = self.pixels_to_mm(diameter_px)
        weight = self.estimate_weight(diameter_mm)
        if diameter_mm < 50:
            return "Small", weight
        elif diameter_mm < 60:
            return "Medium", weight
        else:
            return "Large", weight


class StatisticsLogger:
    def __init__(self, output_path="egg_statistics.csv"):
        self.output_path = output_path
        self.log_file = None
        self.writer = None
        self.initialize_log()
    
    def initialize_log(self):
        try:
            self.log_file = open(self.output_path, 'a', newline='')
            self.writer = csv.DictWriter(
                self.log_file,
                fieldnames=['timestamp', 'egg_id', 'x1', 'y1', 'x2', 'y2', 
                           'confidence', 'class', 'diameter_px', 'diameter_mm', 
                           'size_category', 'weight_g', 'grade']
            )
            self.log_file.seek(0, 2)
            if self.log_file.tell() == 0:
                self.writer.writeheader()
        except Exception as e:
            print(f"Warning: Could not initialize statistics log: {e}")
    
    def log_detection(self, egg_id, x1, y1, x2, y2, conf, egg_class, diameter_px, diameter_mm, size_category, weight_g, grade=""):
        try:
            if self.writer:
                self.writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'egg_id': egg_id,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': round(conf, 4),
                    'class': egg_class,
                    'diameter_px': diameter_px,
                    'diameter_mm': round(diameter_mm, 2),
                    'size_category': size_category,
                    'weight_g': weight_g,
                    'grade': grade
                })
                self.log_file.flush()
        except Exception as e:
            print(f"Warning: Could not log detection: {e}")
    
    def close(self):
        if self.log_file:
            self.log_file.close()


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Egg Detection Live View")
    parser.add_argument("--source", type=str, default="0", 
                        help="Camera index (0, 1, ...) or video file path")
    parser.add_argument("--conf", type=float, default=0.75,
                        help="Confidence threshold (default 0.75)")
    parser.add_argument("--no-logging", action="store_true",
                        help="Disable CSV logging")
    parser.add_argument("--cnn-conf", type=float, default=0.6,
                        help="CNN confidence threshold (default 0.6)")
    args = parser.parse_args()

    model_path = "models/egg_detection_finetuned/weights/best.pt"

    print("Loading YOLO model...")
    try:
        model = YOLO(model_path)
        print(f"Model loaded! Classes: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading EggGradingCNN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = EggGradingCNN(num_classes=2)
    cnn_model.load_state_dict(torch.load('models/egg_grader.pth', map_location=device))
    cnn_model.to(device)
    cnn_model.eval()
    cnn_class_names = ["Damaged", "Not Damaged"]
    print(f"EggGradingCNN loaded! Classes: {cnn_class_names}")

    cnn_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {args.source}")
        return
    
    print(f"\n{'='*60}")
    print("Egg Detection Live View")
    print(f"{'='*60}")
    print(f"Controls: Q=Quit, P=Pause, S=Screenshot, C=CNN Filter, +/-=Confidence")
    print(f"Confidence threshold: {args.conf} | CNN confidence: {args.cnn_conf}")
    print(f"{'='*60}\n")

    enable_cnn_filtering = True

    def map_grade(damage_status, size_category):
        if damage_status == "Damaged":
            return "Reject"
        if size_category == "Large":
            return "A"
        elif size_category == "Medium":
            return "B"
        elif size_category == "Small":
            return "C"
        return "N/A"

    tracker = ObjectTracker()
    calibrator = CameraCalibrator()
    logger = None if args.no_logging else StatisticsLogger()

    paused = False
    pause_frame = None
    conf_threshold = args.conf
    egg_count = 0
    counted_ids = set()
    fps = 0
    prev_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and "/" in source:
                    print("Video ended.")
                    break
                continue
        else:
            frame = pause_frame.copy() if pause_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        results = model(frame, conf=conf_threshold)
        
        egg_detected = False
        egg_boxes = []
        egg_confs = []
        egg_classes = []
        
        boxes = results[0].boxes
        cnn_confs = []
        cnn_labels = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else np.ones(len(xyxy))
            
            raw_detections = []
            for i, box in enumerate(xyxy):
                c = float(confs[i])
                if c >= conf_threshold:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    raw_detections.append((x1, y1, x2, y2, c))
                    egg_boxes.append((x1, y1, x2, y2))
                    egg_confs.append(c)
                    
                    egg_crop = frame[y1:y2, x1:x2]
                    if egg_crop.size > 0:
                        input_tensor = cnn_transform(egg_crop).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = cnn_model(input_tensor)
                            probs = F.softmax(output, dim=1)[0]
                            pred = torch.argmax(output, 1).item()
                            cnn_conf = float(probs[pred].cpu().numpy())
                        cnn_label = cnn_class_names[pred]
                        cnn_confs.append(cnn_conf)
                        cnn_labels.append(cnn_label)
                    else:
                        cnn_confs.append(0.0)
                        cnn_labels.append("unknown")
                    
                    egg_classes.append(cnn_labels[-1])
            
            tracked = tracker.update(raw_detections)
            
            if len(tracked) > 0:
                egg_detected = True
                for obj_id, x1, y1, x2, y2, conf, is_new in tracked:
                    if obj_id not in counted_ids and is_new:
                        counted_ids.add(obj_id)
                        egg_count += 1
                    
                    idx = raw_detections.index((x1, y1, x2, y2, conf)) if (x1, y1, x2, y2, conf) in raw_detections else -1
                    cls_label = cnn_labels[idx] if idx >= 0 else "unknown"
                    
                    if logger:
                        diameter_px = max(x2 - x1, y2 - y1)
                        diameter_mm = calibrator.pixels_to_mm(diameter_px)
                        size, weight = calibrator.categorize_egg(diameter_px)
                        grade = map_grade(cls_label, size)
                        logger.log_detection(obj_id, x1, y1, x2, y2, conf, cls_label, diameter_px, diameter_mm, size, weight, grade)

        new_time = time.time()
        elapsed = max(new_time - prev_time, 1e-6)
        fps = 0.9 * fps + 0.1 * (1 / elapsed) if fps > 0 else 1 / elapsed
        prev_time = new_time

        disp_frame = frame.copy()
        
        if egg_detected:
            for i, (x1, y1, x2, y2) in enumerate(egg_boxes):
                cls = egg_classes[i] if i < len(egg_classes) else "unknown"
                cnn_conf = cnn_confs[i] if i < len(cnn_confs) else 0.0
                
                diameter_px = max(x2 - x1, y2 - y1)
                size, weight = calibrator.categorize_egg(diameter_px)
                grade = map_grade(cls, size)
                
                if enable_cnn_filtering and cnn_conf < args.cnn_conf:
                    display_class = "Low Confidence"
                    box_color = (0, 255, 255)
                else:
                    display_class = cls
                    box_color = (0, 255, 0) if cls == "Not Damaged" else (0, 0, 255)
                
                cv2.rectangle(disp_frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"Grade {grade} | {weight}g"
                cv2.putText(disp_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        filter_status = "ON" if enable_cnn_filtering else "OFF"
        info = f"Count: {egg_count} | In-Frame: {len(egg_boxes)} | CNN-Filter: {filter_status} | FPS: {fps:.1f}"
        cv2.putText(disp_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Egg Detection", disp_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                pause_frame = frame.copy()
        elif key == ord('s'):
            fname = f"egg_snapshot_{egg_count}_{int(time.time())}.jpg"
            cv2.imwrite(fname, disp_frame)
            print(f"Saved: {fname}")
        elif key in [ord('+'), ord('=')]:
            conf_threshold = min(0.99, conf_threshold + 0.05)
            print(f"Confidence: {conf_threshold:.2f}")
        elif key == ord('-'):
            conf_threshold = max(0.1, conf_threshold - 0.05)
            print(f"Confidence: {conf_threshold:.2f}")
        elif key == ord('c'):
            enable_cnn_filtering = not enable_cnn_filtering
            print(f"CNN Filter: {'ON' if enable_cnn_filtering else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    
    if logger:
        logger.close()
        print(f"Statistics saved to egg_statistics.csv")
    
    print(f"Session complete. Total eggs counted: {egg_count}")


if __name__ == "__main__":
    main()
