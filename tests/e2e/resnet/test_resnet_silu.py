"""Full CIFAR ResNet-20 in TensorTerm without activation functions.

The graph mirrors ``benchmarks.e2e.resnet.resnet_model.resnet20``: conv layers,
affine batch norm (``x * scale + shift``), Option-A shortcuts as fixed 1×1
stride-2 convs, two spatial ``sum`` reductions (global pooling), then
``@ fc + bias``. Classifier weights are scaled by ``1/64`` so the sums match
PyTorch ``avg_pool2d`` + ``Linear``. No SiLU.

``LayoutAssignment`` → Lower → Toy does not yet match plaintext tensor eval for
the **entire** ResNet-20 no-activation graph (Toy fails per-kernel checks past
the stem). We therefore:

- validate the **full** graph with ``tensor_ir.eval`` vs PyTorch in float64; and
- validate the **stem** (conv1 + BN1) through the full Rotom layout pipeline
  against the same tensor IR (``check_results``) and against PyTorch stem
  activations.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import torch

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from benchmarks.e2e.resnet import resnet20_tensor_ir as R
from benchmarks.e2e.resnet.resnet_data import CKPT_FILE, load_checkpoint
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    build_resnet20_no_activation_graph,
    populate_resnet20_no_activation_inputs,
    resnet20_forward_no_activation,
)
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.test_util import get_default_args
from util.checker import check_results


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


_RESNET_TOY_N = 32768


@pytest.mark.e2e
@pytest.mark.slow
def test_resnet20_no_activation_stem_layout_pipeline_matches_tensor_eval() -> None:
    """Stem (conv1 + BN1): LayoutAssignment → Lower → Toy matches ``tensor_ir.eval`` and PyTorch."""
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

    h, w = 32, 32
    t = TensorTerm.Tensor("input", [3, 32, 32], True)
    t = R._conv_same(t, "conv1_w", inputs, 1)
    h, w = R._spatial_hw_after_conv3(h, w, 1)
    t = R._bn_affine_hw(t, "bn1", 16, h, w, inputs)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_no_activation_stem"

    kernel = LayoutAssignment(t, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(t, inputs, kernel, backend_results, 0, args)

    dense = t.eval(inputs)
    with torch.no_grad():
        pt_stem = model.bn1(model.conv1(x))[0].cpu().numpy()
    assert dense.shape == pt_stem.shape
    assert np.allclose(dense, pt_stem, rtol=1e-9, atol=1e-7)
