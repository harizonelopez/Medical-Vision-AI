
from pathlib import Path
from torchvision.datasets import CIFAR10

# Store CIFAR-10 separately from the chest X-ray dataset
DATA_DIR = Path(__file__).resolve().parent / "data" / "external"

print("Downloading CIFAR-10...")

train_data = CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=True
)

test_data = CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=True
)

print("\nDownload completed!")
print("Training images:", len(train_data))
print("Testing images:", len(test_data))
print("Image classes:", train_data.classes)
