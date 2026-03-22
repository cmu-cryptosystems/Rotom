"""CIFAR ResNet-20 TensorTerm e2e tests (SiLU checkpoint, affine BN in IR).

Stem layout test: conv + BN TensorTerm (no SiLU in the small graph) vs TensorEvaluator
and PyTorch ``bn1(conv1(x))`` for numerical agreement.
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
from benchmarks.e2e.resnet.resnet20_tensor_ir import populate_resnet20_inputs
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


_RESNET_TOY_N = 32768


@pytest.mark.e2e
@pytest.mark.slow
def test_resnet20_stem_matches_tensor_eval() -> None:
    """Stem (conv1 + BN1): LayoutAssignment → Lower → Toy matches ``tensor_ir.eval`` and PyTorch."""
    if not os.path.exists(CKPT_FILE):
        pytest.skip(f"Missing checkpoint at {CKPT_FILE}")

    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    _load_dacapo_weights(model, CKPT_FILE)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)

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
    args.benchmark = "resnet20_stem"

    kernel = LayoutAssignment(t, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(t, inputs, kernel, backend_results, 0, args)

    dense = t.eval(inputs)
    with torch.no_grad():
        pt_stem = model.bn1(model.conv1(x))[0].cpu().numpy()
    assert dense.shape == pt_stem.shape
    assert np.allclose(dense, pt_stem, rtol=1e-9, atol=1e-7)
