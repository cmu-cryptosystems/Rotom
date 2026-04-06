#!/usr/bin/env python3
"""cProfile LayoutAssignment → Lower → Toy for stem+layer1 conv-only IR.

Default graphs the same three convolutions as ``test_resnet20_stem_plus_layer1_first_conv_toy``
(stem + ``l1_0`` conv1 + conv2). Set ``ROTOM_PROFILE_CONVS=7`` to profile the full stem + layer1
chain (needs large RAM at ``n=32768``).

Run from repo root::

    python benchmarks/e2e/resnet/profile_stem_layer1_convs.py
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from assignment.assignment import LayoutAssignment  # noqa: E402
from backends.toy import Toy  # noqa: E402
from benchmarks.e2e.resnet.resnet_model import resnet20  # noqa: E402
from benchmarks.e2e.resnet.resnet20_tensor_ir import (  # noqa: E402
    build_resnet20_stem_plus_layer1_convs_only,
    populate_resnet20_inputs,
)
from frontends.tensor import TensorTerm  # noqa: E402
from lower.lower import Lower  # noqa: E402
from tests.test_util import get_default_args  # noqa: E402

_RESNET20_CONV_N = 32768


def _build_inputs_and_ir_three() -> tuple[object, dict]:
    model = resnet20()
    w_stem = model.conv1.weight.detach().cpu().numpy().astype(np.float64)
    w_l10_c1 = model.layer1[0].conv1.weight.detach().cpu().numpy().astype(np.float64)
    w_l10_c2 = model.layer1[0].conv2.weight.detach().cpu().numpy().astype(np.float64)
    torch.manual_seed(0)
    x_np = (
        torch.randn(1, 3, 32, 32, dtype=torch.float32)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    inp = TensorTerm.Tensor("input", [3, 32, 32], True)
    h = TensorTerm.conv2d(
        inp, TensorTerm.Tensor("conv1_w", list(w_stem.shape), False), 1, "same"
    )
    h = TensorTerm.conv2d(
        h, TensorTerm.Tensor("l1_0_conv1_w", list(w_l10_c1.shape), False), 1, "same"
    )
    tensor_ir = TensorTerm.conv2d(
        h, TensorTerm.Tensor("l1_0_conv2_w", list(w_l10_c2.shape), False), 1, "same"
    )
    inputs = {
        "input": x_np,
        "conv1_w": w_stem,
        "l1_0_conv1_w": w_l10_c1,
        "l1_0_conv2_w": w_l10_c2,
    }
    return tensor_ir, inputs


def _build_inputs_and_ir_seven() -> tuple[object, dict]:
    model = resnet20()
    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    torch.manual_seed(0)
    inputs["input"] = (
        torch.randn(1, 3, 32, 32, dtype=torch.float32)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    tensor_ir = build_resnet20_stem_plus_layer1_convs_only(inputs)
    return tensor_ir, inputs


def main() -> None:
    nconv = int(os.environ.get("ROTOM_PROFILE_CONVS", "3"))
    if nconv == 7:
        tensor_ir, inputs = _build_inputs_and_ir_seven()
        benchmark = "resnet20_conv_stem_layer1_all"
    elif nconv == 3:
        tensor_ir, inputs = _build_inputs_and_ir_three()
        benchmark = "resnet20_conv_stem_l10_both"
    else:
        raise SystemExit("ROTOM_PROFILE_CONVS must be 3 or 7")

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET20_CONV_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = benchmark
    args.skip_toy_eval_checks = True

    def run() -> None:
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        Toy(circuit_ir, inputs, args).run()

    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()

    for sort, title in (
        ("cumulative", "By cumulative time"),
        ("tottime", "By tottime"),
    ):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(sort).print_stats(35)
        print(title)
        print(s.getvalue())


if __name__ == "__main__":
    main()
