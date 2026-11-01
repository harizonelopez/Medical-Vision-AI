from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18


# --------------------------------
# Configuration
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xray_detector.pt"

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.0001

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------
# Image transformations
# --------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------
# Combined dataset
# --------------------------------

class ImageValidatorDataset(Dataset):

    def __init__(self, xray_paths, cifar_images, transform=None):

        self.transform = transform

        # Label 1: Chest X-ray
        self.samples = [
            (path, 1) for path in xray_paths
        ]

        # Label 0: Irrelevant image
        self.samples.extend([
            (image, 0) for image in cifar_images
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        source, label = self.samples[index]

        if isinstance(source, Path):
            with Image.open(source) as image:
                image = image.convert("RGB")
        else:
            image = Image.fromarray(source)

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------------
# Load training images
# --------------------------------

def load_xray_paths():

    normal_dir = DATA_DIR / "normal"
    pneumonia_dir = DATA_DIR / "pneumonia"

    if not normal_dir.exists() or not pneumonia_dir.exists():
        raise FileNotFoundError(
            "Expected data/normal and data/pneumonia folders."
        )

    extensions = {".png", ".jpg", ".jpeg"}

    normal_images = [
        path for path in normal_dir.rglob("*")
        if path.suffix.lower() in extensions
    ]

    pneumonia_images = [
        path for path in pneumonia_dir.rglob("*")
        if path.suffix.lower() in extensions
    ]

    xray_paths = normal_images + pneumonia_images

    if not xray_paths:
        raise RuntimeError("No chest X-ray images were found.")

    print("Normal X-rays:", len(normal_images))
    print("Pneumonia X-rays:", len(pneumonia_images))
    print("Total X-rays:", len(xray_paths))

    return xray_paths


# --------------------------------
# Load CIFAR-10
# --------------------------------

def load_cifar_images(number_of_xrays):

    print("\nLoading CIFAR-10...")

    cifar = CIFAR10(
        root=str(DATA_DIR / "external"),
        train=True,
        download=False,
    )

    # Keep the negative class balanced with the X-ray class.
    random.seed(42)

    selected_indices = random.sample(
        range(len(cifar.data)),
        min(number_of_xrays, len(cifar.data)),
    )

    cifar_images = [
        cifar.data[index]
        for index in selected_indices
    ]

    print("CIFAR-10 images selected:", len(cifar_images))

    return cifar_images


# --------------------------------
# Evaluation
# --------------------------------

def evaluate_model(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(DEVICE)

            logits = model(images)
            scores = torch.sigmoid(logits).squeeze(1)

            predictions = (scores >= 0.5).int()

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    precision = precision_score(
        all_labels, all_predictions, zero_division=0
    )

    recall = recall_score(
        all_labels, all_predictions, zero_division=0
    )

    auc = roc_auc_score(all_labels, all_scores)

    matrix = confusion_matrix(
        all_labels, all_predictions
    )

    return accuracy, precision, recall, auc, matrix


# --------------------------------
# Training
# --------------------------------

def main():

    print("Device:", DEVICE)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    xray_paths = load_xray_paths()

    cifar_images = load_cifar_images(len(xray_paths))

    dataset = ImageValidatorDataset(
        xray_paths,
        cifar_images,
    )

    labels = np.array([
        label for _, label in dataset.samples
    ])

    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    train_dataset = ImageValidatorDataset(
        xray_paths,
        cifar_images,
        transform=train_transform,
    )

    val_dataset = ImageValidatorDataset(
        xray_paths,
        cifar_images,
        transform=validation_transform,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("\nTraining images:", len(train_indices))
    print("Validation images:", len(val_indices))

    # Load pretrained ResNet-18 dataset.
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Freeze earlier feature extraction layers.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the final convolutional block.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # Replace the original classification layer.
    model.fc = nn.Linear(
        model.fc.in_features,
        1,
    )

    model = model.to(DEVICE)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    best_auc = 0.0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)

            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = loss_function(logits, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        accuracy, precision, recall, auc, matrix = (
            evaluate_model(model, val_loader)
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Training loss: {average_loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation precision: {precision:.4f}")
        print(f"Validation recall: {recall:.4f}")
        print(f"Validation ROC-AUC: {auc:.4f}")
        print("Confusion matrix:")
        print(matrix)

        if auc > best_auc:

            best_auc = auc

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print("Saved improved validator model.")

    print("\nTraining completed.")
    print("Best validation AUC:", round(best_auc, 4))
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()