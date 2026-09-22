import cv2
import torch
import yaml

from model import EggGradingCNN
from utils import get_transforms


def infer(image_path, model_path, config_path):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Image preprocessing
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_transforms(train=False, img_size=cfg["data"]["img_size"])
    tensor = transform(image).unsqueeze(0)
    # Load model
    model = EggGradingCNN(num_classes=cfg["model"]["num_classes"])
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, 1).item()
    return pred


if __name__ == "__main__":
    pred = infer(
        "data/processed/sample.png", "models/egg_grader.pth", "config/config.yaml"
    )
    print(f"Prediction: {pred}")
