import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.e2e.mnist.mnist_data import MODEL_FILE, load_mnist_test_set


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
    images, labels = load_mnist_test_set()
    accuracy = _evaluate_traced_model(MODEL_FILE, images, labels)
    print(f"Traced MNIST model accuracy on test set: {accuracy:.4%}")


if __name__ == "__main__":
    main()
