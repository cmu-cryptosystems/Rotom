"""Plaintext (PyTorch) e2e test for DaCapo ResNet20 (SiLU) checkpoint on CIFAR-10."""

from __future__ import annotations

import os

import pytest
import torch

from benchmarks.e2e.resnet.resnet_data import (
    CKPT_FILE,
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
@pytest.mark.slow
def test_resnet20_silu_plaintext_checkpoint_accuracy_smoke():
    """
    Smoke-test accuracy evaluation in plaintext (PyTorch).

    Skips if required local artifacts are missing:
    - Checkpoint: `benchmarks/e2e/resnet/data/resnet20.silu.model`
    """
    if not os.path.exists(CKPT_FILE):
        pytest.skip(f"Missing checkpoint at {CKPT_FILE}")

    device = _device()
    model = resnet20(num_classes=10)
    load_dacapo_checkpoint_into(model, CKPT_FILE)
    model.to(device)

    try:
        loader = cifar10_test_loader(batch_size=256, num_workers=0)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    acc = accuracy_top1(model, loader, device=device)

    assert 0.0 <= acc <= 1.0
    min_acc = float(os.environ.get("ROTOM_RESNET_MIN_ACC", "0.70"))
    assert acc >= min_acc
