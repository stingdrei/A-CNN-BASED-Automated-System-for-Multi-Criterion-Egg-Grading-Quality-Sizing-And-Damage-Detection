"""Run the YOLO detector on a local camera without the web application."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="egg_detection/train1/weights/best.pt",
        help="Path to a YOLO checkpoint",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--device", default=None, help="Inference device, e.g. mps or cuda:0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check camera permissions or use --camera 1."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    model = YOLO(str(model_path))
    class_names = model.names
    window_name = "Egg detector - press q or Esc to quit"
    print(f"Camera {args.camera} started. Press q or Esc to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Camera returned an unreadable frame")

            prediction_args = {"source": frame, "conf": args.confidence, "verbose": False}
            if args.device:
                prediction_args["device"] = args.device
            result = model.predict(**prediction_args)[0]
            annotated = result.plot()
            count = len(result.boxes) if result.boxes is not None else 0
            cv2.putText(
                annotated,
                f"Eggs detected: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
