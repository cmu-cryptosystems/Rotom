"""Full CIFAR ResNet-20 in TensorTerm without activation functions.

The graph mirrors ``benchmarks.e2e.resnet.resnet_model.resnet20``: conv layers,
affine batch norm (``x * scale + shift``), Option-A shortcuts as fixed 1×1
stride-2 convs, two spatial ``sum`` reductions (global pooling), then
``@ fc + bias``. Classifier weights are scaled by ``1/64`` so the sums match
PyTorch ``avg_pool2d`` + ``Linear``. No SiLU.

``LayoutAssignment`` does not yet produce kernels for every op in this full
32×32 graph; we validate ``tensor_ir.eval`` against PyTorch in float64.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import torch

from benchmarks.e2e.resnet.resnet_data import CKPT_FILE, load_checkpoint
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    build_resnet20_no_activation_graph,
    populate_resnet20_no_activation_inputs,
    resnet20_forward_no_activation,
)


def _load_dacapo_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None or not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise FileNotFoundError(
            f"Missing/invalid DaCapo checkpoint at {ckpt_path} (need dict with 'state_dict')."
        )
    state: Any = ckpt["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected 'state_dict' type: {type(state)}")
    if any(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)


@pytest.mark.e2e
@pytest.mark.slow
def test_resnet20_no_activation_eval_matches_pytorch() -> None:
    """Full ResNet-20 TensorTerm graph (no activations) matches PyTorch in float64."""
    if not os.path.exists(CKPT_FILE):
        pytest.skip(f"Missing checkpoint at {CKPT_FILE}")

    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    _load_dacapo_weights(model, CKPT_FILE)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_resnet20_no_activation_inputs(model, inputs)

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()

    tensor_ir = build_resnet20_no_activation_graph(inputs)

    with torch.no_grad():
        pt_logits = resnet20_forward_no_activation(model, x).cpu().numpy()

    dense = tensor_ir.eval(inputs)
    # TensorEvaluator pads the classifier axis to the next power of two (10 -> 16).
    rot_logits = dense[:, :10]

    assert rot_logits.shape == (1, 10)
    assert np.allclose(rot_logits, pt_logits, rtol=1e-9, atol=1e-7)
