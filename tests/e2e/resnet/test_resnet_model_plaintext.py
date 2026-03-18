"""Plaintext (PyTorch) e2e test for DaCapo ResNet20 (SiLU) checkpoint on CIFAR-10."""

from __future__ import annotations

import os

import pytest
import torch

from benchmarks.e2e.resnet.resnet_data import (
    CKPT_FILE,
    DATA_DIR,
    accuracy_top1,
    cifar10_test_loader,
    load_checkpoint,
)
from benchmarks.e2e.resnet.resnet_model import resnet20


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dacapo_checkpoint_into(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None or "state_dict" not in ckpt:
        raise FileNotFoundError(
            f"Missing DaCapo checkpoint at {ckpt_path}. "
            f"Expected a torch.load()-able dict with key 'state_dict'."
        )
    state = ckpt["state_dict"]
    # DaCapo saves DataParallel state_dict keys prefixed with "module.".
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)


@pytest.mark.e2e
def test_resnet20_silu_plaintext_checkpoint_accuracy_smoke():
    """
    Smoke-test accuracy evaluation in plaintext (PyTorch).

    Skips if required local artifacts are missing:
    - CIFAR-10 tarball: `benchmarks/e2e/resnet/data/cifar-10-python.tar.gz`
    - Checkpoint: `benchmarks/e2e/resnet/data/resnet20.silu.model`
    """

    cifar_tar = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")

    if not os.path.exists(cifar_tar):
        pytest.skip(f"Missing CIFAR-10 archive at {cifar_tar}")
    if not os.path.exists(CKPT_FILE):
        pytest.skip(f"Missing checkpoint at {CKPT_FILE}")

    device = _device()
    model = resnet20(num_classes=10)
    load_dacapo_checkpoint_into(model, CKPT_FILE)
    model.to(device)

    loader = cifar10_test_loader(batch_size=256, num_workers=0)
    acc = accuracy_top1(model, loader, device=device)

    assert 0.0 <= acc <= 1.0
    # Very lenient lower bound to catch obvious checkpoint/model mismatches.
    assert acc >= 0.70
