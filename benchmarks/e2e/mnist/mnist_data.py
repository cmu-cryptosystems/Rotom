"""Shared MNIST data utilities for benchmarks and tests.

This module centralizes paths and loaders for the MNIST test set and the
TorchScript model used by the MNIST benchmarks/tests.
"""

import os
import struct
from typing import Dict, Tuple

import numpy as np
import torch

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
    "Load MNIST labels from IDX1 ubyte file into a long tensor [N]."
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


def load_mnist_test_set() -> Tuple[torch.Tensor, torch.Tensor]:
    "Load MNIST test images and labels from the shared data directory."
    images = _load_idx_images(IMAGES_FILE)
    labels = _load_idx_labels(LABELS_FILE)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mismatch between images ({images.shape[0]}) and labels "
            f"({labels.shape[0]})"
        )
    return images, labels


def extract_traced_mnist_linears(model_path: str) -> Dict[str, np.ndarray]:
    """Load TorchScript MNIST model and recover its two Linear layers from state_dict.

    The traced model may not expose Python nn.Linear modules directly (it is
    typically a RecursiveScriptModule), so we instead inspect its parameters
    and look for 2D weight tensors with shapes matching the expected MNIST
    architecture:
        - fc1: weight [hidden_dim, 784]
        - fc2: weight [10, hidden_dim]
    """
    ts_model = torch.jit.load(model_path, map_location="cpu")
    core = getattr(ts_model, "model", ts_model)
    state = core.state_dict()

    # Collect all 2D weight tensors.
    weight_tensors = {
        name: tensor
        for name, tensor in state.items()
        if name.endswith("weight") and tensor.ndim == 2
    }
    if not weight_tensors:
        raise RuntimeError("No 2D weight tensors found in traced model state_dict")

    # Heuristic: fc1 has in_features == 784; pick the one with largest out_features.
    fc1_name, fc1_weight = None, None
    for name, w in weight_tensors.items():
        out_f, in_f = w.shape
        if in_f == 784:
            if fc1_weight is None or out_f > fc1_weight.shape[0]:
                fc1_name, fc1_weight = name, w
    if fc1_weight is None:
        raise RuntimeError("Could not find fc1 weight with in_features == 784")

    hidden_dim = fc1_weight.shape[0]
    in_dim = fc1_weight.shape[1]

    # Heuristic: fc2 has out_features == 10 and in_features == hidden_dim.
    fc2_name, fc2_weight = None, None
    for name, w in weight_tensors.items():
        out_f, in_f = w.shape
        if out_f == 10 and in_f == hidden_dim:
            fc2_name, fc2_weight = name, w
            break
    if fc2_weight is None:
        raise RuntimeError(
            f"Could not find fc2 weight with out_features == 10 and in_features == {hidden_dim}"
        )

    # Fetch corresponding biases.
    def _bias_for(weight_name: str):
        bias_name = weight_name.replace("weight", "bias")
        if bias_name not in state:
            raise RuntimeError(f"Missing bias parameter for {weight_name}")
        return state[bias_name]

    fc1_bias = _bias_for(fc1_name)
    fc2_bias = _bias_for(fc2_name)

    out_dim = fc2_weight.shape[0]

    return {
        "in_dim": int(in_dim),
        "hidden_dim": int(hidden_dim),
        "out_dim": int(out_dim),
        "fc1_w": fc1_weight.detach().cpu().numpy(),  # [hidden_dim, in_dim]
        "fc1_b": fc1_bias.detach().cpu().numpy(),  # [hidden_dim]
        "fc2_w": fc2_weight.detach().cpu().numpy(),  # [out_dim, hidden_dim]
        "fc2_b": fc2_bias.detach().cpu().numpy(),  # [out_dim]
    }
