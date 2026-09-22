"""
Capture egg images from webcam for YOLO training.
Usage: python capture_eggs.py
Press SPACE to capture, Q to quit.
"""
import cv2
import os
from datetime import datetime

OUTPUT_DIR = "data/detection/images/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    exit()

count = 0
print("Press SPACE to capture, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.putText(frame, f"Images: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Capture Eggs", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:
        fname = f"egg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), frame)
        count += 1
        print(f"Saved {fname}")

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images to {OUTPUT_DIR}")