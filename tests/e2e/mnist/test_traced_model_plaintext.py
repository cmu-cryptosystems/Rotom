import os
import struct
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_FILE = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
LABELS_FILE = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")
MODEL_FILE = os.path.join(DATA_DIR, "traced_model.pt")


def _load_idx_images(path: str) -> torch.Tensor:
    """Load MNIST images from IDX3 ubyte file into a float32 tensor [N, 1, 28, 28]."""
    with open(path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"Incomplete IDX image header in {path}")
        magic, num_images, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(
                f"Unexpected magic number {magic} in {path}, expected 2051"
            )
        data = f.read()

    images = np.frombuffer(data, dtype=np.uint8)
    expected_pixels = num_images * rows * cols
    if images.size != expected_pixels:
        raise ValueError(
            f"Image file {path} has {images.size} pixels, expected {expected_pixels}"
        )

    images = images.reshape(num_images, rows, cols).astype(np.float32) / 255.0
    images = np.expand_dims(images, 1)  # [N, 1, 28, 28]
    return torch.from_numpy(images)


def _load_idx_labels(path: str) -> torch.Tensor:
    """Load MNIST labels from IDX1 ubyte file into a long tensor [N]."""
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Incomplete IDX label header in {path}")
        magic, num_items = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(
                f"Unexpected magic number {magic} in {path}, expected 2049"
            )
        data = f.read()

    labels = np.frombuffer(data, dtype=np.uint8)
    if labels.size != num_items:
        raise ValueError(
            f"Label file {path} has {labels.size} labels, expected {num_items}"
        )
    return torch.from_numpy(labels.astype(np.int64))


def _load_mnist_test_set() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST test images and labels from the local data directory."""
    images = _load_idx_images(IMAGES_FILE)
    labels = _load_idx_labels(LABELS_FILE)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mismatch between images ({images.shape[0]}) and labels "
            f"({labels.shape[0]})"
        )
    return images, labels


def _evaluate_traced_model(
    model_path: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 128,
) -> float:
    """Run accuracy evaluation for a TorchScript MNIST model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # The traced model is an MLP with a first Linear(784, 512) layer.
            # It expects each sample as a flat 784-dimensional vector, so we
            # flatten the 1x28x28 images before passing them to the model.
            if x.ndim > 2:
                x = x.view(x.size(0), -1)
            y = y.to(device)
            logits = model(x)

            # Allow the traced model to return extra dimensions (e.g. flattened logits).
            if logits.ndim > 2:
                logits = logits.view(logits.size(0), -1)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    if total == 0:
        raise ValueError("No samples loaded from MNIST test set.")

    return correct / total


def main() -> None:
    images, labels = _load_mnist_test_set()
    accuracy = _evaluate_traced_model(MODEL_FILE, images, labels)
    print(f"Traced MNIST model accuracy on test set: {accuracy:.4%}")


if __name__ == "__main__":
    main()
