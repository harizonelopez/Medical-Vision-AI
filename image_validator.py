from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xray_detector.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224
X_RAY_THRESHOLD = 0.5


def load_validator():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chest X-ray detector has not been trained yet."
        )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


def validate_image(image: Image.Image, model):

    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        xray_score = torch.sigmoid(logits).item()

    is_xray = xray_score >= X_RAY_THRESHOLD

    return is_xray, xray_score