"""Shared CIFAR-10 utilities for ResNet e2e benchmark.

This is intentionally minimal:
- load CIFAR-10 test set from `data/cifar-10-python.tar.gz` (copied from DaCapo)
- load DaCapo's `resnet20.silu.model` checkpoint
"""

import os
import pickle
import tarfile
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_FILE = os.path.join(DATA_DIR, "resnet20.silu.model")

_ARCHIVE_PATH = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
_EXTRACTED_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


def ensure_cifar10_extracted() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.isdir(_EXTRACTED_DIR):
        return
    if not os.path.exists(_ARCHIVE_PATH):
        if (
            os.environ.get("ROTOM_OFFLINE", "").strip()
            or os.environ.get("CI", "").strip()
        ):
            raise FileNotFoundError(
                f"Missing CIFAR-10 archive at {_ARCHIVE_PATH}. "
                "Set ROTOM_OFFLINE=0 to allow auto-download, or copy it from "
                "DaCapo: dacapo/examples/data/CIFAR10/cifar-10-python.tar.gz"
            )
        try:
            from torchvision.datasets.utils import download_url
        except Exception as e:  # pragma: no cover
            raise FileNotFoundError(
                f"Missing CIFAR-10 archive at {_ARCHIVE_PATH}. "
                "Install torchvision to enable auto-download, or copy it from "
                "DaCapo: dacapo/examples/data/CIFAR10/cifar-10-python.tar.gz"
            ) from e

        download_url(
            _CIFAR10_URL,
            root=DATA_DIR,
            filename=os.path.basename(_ARCHIVE_PATH),
            md5=_CIFAR10_MD5,
        )
    with tarfile.open(_ARCHIVE_PATH, "r:gz") as tf:
        tf.extractall(path=DATA_DIR)


def _load_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"]  # [N, 3072], uint8
    labels = np.array(d[b"labels"], dtype=np.int64)  # [N]
    x = data.reshape(-1, 3, 32, 32)
    return x, labels


class CIFAR10Numpy(Dataset):
    def __init__(self, train: bool):
        ensure_cifar10_extracted()
        if train:
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = ["test_batch"]

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for bf in batch_files:
            x, y = _load_batch(os.path.join(_EXTRACTED_DIR, bf))
            xs.append(x)
            ys.append(y)

        self.x = np.concatenate(xs, axis=0)  # uint8 [N, 3, 32, 32]
        self.y = np.concatenate(ys, axis=0)  # int64 [N]

        # DaCapo uses ImageNet-style normalization for CIFAR10 in its script.
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
            3, 1, 1
        )
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
            3, 1, 1
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.x[idx]).to(torch.float32) / 255.0  # [3,32,32]
        img = (img - self.mean) / self.std
        label = int(self.y[idx])
        return img, label


def cifar10_test_loader(batch_size: int = 256, num_workers: int = 2) -> DataLoader:
    ds = CIFAR10Numpy(train=False)
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
