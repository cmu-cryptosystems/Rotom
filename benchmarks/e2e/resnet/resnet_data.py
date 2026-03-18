"""Shared CIFAR-10 utilities for ResNet e2e benchmark.

- load CIFAR-10 test set via TorchVision's downloader
- load DaCapo's `resnet20.silu.model` checkpoint
"""

import os
from typing import Any

import torch
from torch.utils.data import DataLoader


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_FILE = os.path.join(DATA_DIR, "resnet20.silu.model")


def cifar10_test_dataset() -> Any:
    os.makedirs(DATA_DIR, exist_ok=True)
    offline = (
        os.environ.get("ROTOM_OFFLINE", "").strip() or os.environ.get("CI", "").strip()
    )

    try:
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import Compose, Normalize, ToTensor
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "torchvision is required to load/download CIFAR-10. "
            "Install it (see requirements.txt)."
        ) from e

    # DaCapo uses ImageNet-style normalization for CIFAR10 in its script.
    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    try:
        return CIFAR10(
            root=DATA_DIR, train=False, download=not offline, transform=transform
        )
    except RuntimeError as e:
        # TorchVision throws RuntimeError when the dataset isn't found and download=False.
        raise FileNotFoundError(
            f"Missing CIFAR-10 dataset under {DATA_DIR}. "
            "Unset CI/ROTOM_OFFLINE (or set ROTOM_OFFLINE=0) to allow auto-download."
        ) from e


def cifar10_test_loader(batch_size: int = 256, num_workers: int = 2) -> DataLoader:
    ds = cifar10_test_dataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_checkpoint(path: str = CKPT_FILE):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def save_checkpoint(state: dict, path: str = CKPT_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


@torch.no_grad()
def accuracy_top1(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = torch.as_tensor(labels, device=device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)
