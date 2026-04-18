"""TensorTerm IR for AlexNetSmall (SiLU via ``poly_call('silu', ...)``).

Depth checkpoints (see :func:`build_alexnet_small_silu_poly_graph_to_depth`):

- ``stem``: conv1 + affine BN + SiLU poly (``[64, 32, 32]``).
- ``stage2``: stem + conv2 block (``[128, 16, 16]``).
- ``stage3``: through conv3 block (``[256, 8, 8]``).
- ``full``: conv backbone + global sum-pool + two-layer MLP to logits.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch.nn as nn

from frontends.tensor import TensorTerm
from util.silu_polycall_eval import (
    DEFAULT_SILU_POLY_DEGREE,
    DEFAULT_SILU_POLY_NODES,
    SILU_POLY_DEGREE_KEY,
    SILU_POLY_NODES_KEY,
)


def _fill_conv_weights(inputs: dict, key: str, conv: nn.Conv2d) -> None:
    inputs[key] = conv.weight.detach().cpu().numpy().astype(np.float64)


def _fill_bn_affine(inputs: dict, prefix: str, bn: nn.BatchNorm2d) -> None:
    mean = bn.running_mean.detach().cpu().numpy()
    var = bn.running_var.detach().cpu().numpy()
    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    eps = float(bn.eps)
    std = np.sqrt(var + eps)
    scale = (gamma / std).astype(np.float64).reshape(-1, 1, 1)
    shift = (beta - mean * gamma / std).astype(np.float64).reshape(-1, 1, 1)
    inputs[f"{prefix}_scale"] = scale
    inputs[f"{prefix}_shift"] = shift


def _w_tensor(inputs: dict, key: str) -> TensorTerm:
    arr = inputs[key]
    return TensorTerm.Tensor(key, list(arr.shape), False)


def _spatial_hw_after_conv3(h: int, w: int, stride: int) -> tuple[int, int]:
    p = 1
    k = 3
    h_pad = h + 2 * p
    w_pad = w + 2 * p
    return (h_pad - k) // stride + 1, (w_pad - k) // stride + 1


def _bn_affine_hw(
    x: TensorTerm, prefix: str, num_ch: int, h: int, w: int, inputs: dict
) -> TensorTerm:
    sk = f"{prefix}_scale_hw_{h}_{w}"
    shk = f"{prefix}_shift_hw_{h}_{w}"
    if sk not in inputs:
        inputs[sk] = np.broadcast_to(inputs[f"{prefix}_scale"], (num_ch, h, w)).astype(
            np.float64
        )
    if shk not in inputs:
        inputs[shk] = np.broadcast_to(inputs[f"{prefix}_shift"], (num_ch, h, w)).astype(
            np.float64
        )
    sc = TensorTerm.Tensor(sk, [num_ch, h, w], False)
    shift = TensorTerm.Tensor(shk, [num_ch, h, w], False)
    return x * sc + shift


def _conv_same(x: TensorTerm, weight_key: str, inputs: dict, stride: int) -> TensorTerm:
    return TensorTerm.conv2d(x, _w_tensor(inputs, weight_key), stride, "same")


def _silu_poly(x: TensorTerm) -> TensorTerm:
    return x.poly_call("silu", -20.0, 20.0)


def populate_alexnet_small_inputs(model: nn.Module, inputs: dict) -> None:
    """Populate all plaintext tensors needed by AlexNetSmall TensorIR builders."""
    inputs.clear()
    for i in range(1, 6):
        conv = getattr(model, f"conv{i}")
        bn = getattr(model, f"bn{i}")
        _fill_conv_weights(inputs, f"conv{i}_w", conv)
        _fill_bn_affine(inputs, f"bn{i}", bn)

    # We implement avg-pool as sum over 8x8, so scale fc1 by 1/64.
    w1 = model.fc1.weight.detach().cpu().numpy()
    b1 = model.fc1.bias.detach().cpu().numpy()
    inputs["fc1"] = (w1.T / 64.0).astype(np.float64)
    inputs["fc1_b"] = b1.reshape(1, -1).astype(np.float64)

    w2 = model.fc2.weight.detach().cpu().numpy()
    b2 = model.fc2.bias.detach().cpu().numpy()
    inputs["fc2"] = w2.T.astype(np.float64)
    inputs["fc2_b"] = b2.reshape(1, -1).astype(np.float64)

    inputs[SILU_POLY_DEGREE_KEY] = DEFAULT_SILU_POLY_DEGREE
    inputs[SILU_POLY_NODES_KEY] = DEFAULT_SILU_POLY_NODES


AlexnetSmallDepth = Literal["stem", "stage2", "stage3", "full"]


def build_alexnet_small_silu_poly_graph_to_depth(
    inputs: dict, depth: AlexnetSmallDepth
) -> TensorTerm:
    """AlexNetSmall SiLU-poly prefix: stop after ``stem``, ``stage2``, ``stage3``, or ``full``."""
    h, w = 32, 32
    x = TensorTerm.Tensor("input", [3, 32, 32], True)

    x = _conv_same(x, "conv1_w", inputs, 1)
    h, w = _spatial_hw_after_conv3(h, w, 1)
    x = _bn_affine_hw(x, "bn1", 64, h, w, inputs)
    x = _silu_poly(x)
    if depth == "stem":
        return x

    x = _conv_same(x, "conv2_w", inputs, 2)
    h, w = _spatial_hw_after_conv3(h, w, 2)
    x = _bn_affine_hw(x, "bn2", 128, h, w, inputs)
    x = _silu_poly(x)
    if depth == "stage2":
        return x

    x = _conv_same(x, "conv3_w", inputs, 2)
    h, w = _spatial_hw_after_conv3(h, w, 2)
    x = _bn_affine_hw(x, "bn3", 256, h, w, inputs)
    x = _silu_poly(x)
    if depth == "stage3":
        return x

    x = _conv_same(x, "conv4_w", inputs, 1)
    h, w = _spatial_hw_after_conv3(h, w, 1)
    x = _bn_affine_hw(x, "bn4", 256, h, w, inputs)
    x = _silu_poly(x)

    x = _conv_same(x, "conv5_w", inputs, 1)
    h, w = _spatial_hw_after_conv3(h, w, 1)
    x = _bn_affine_hw(x, "bn5", 128, h, w, inputs)
    x = _silu_poly(x)

    # Global average pool via two sums and fc1 scaled by 1/(h*w)=1/64.
    x = x.sum(1)
    x = x.sum(1)
    x = x.reshape(0, {0: 1, 1: 128})

    x = x @ _w_tensor(inputs, "fc1") + TensorTerm.Tensor("fc1_b", [1, 64], False)
    x = _silu_poly(x)
    x = x @ _w_tensor(inputs, "fc2") + TensorTerm.Tensor("fc2_b", [1, 10], False)
    return x


def build_alexnet_small_silu_poly_graph_through_stage2(inputs: dict) -> TensorTerm:
    """Prefix: conv1/bn1/silu -> conv2/bn2/silu (same as ``depth="stage2"``)."""
    return build_alexnet_small_silu_poly_graph_to_depth(inputs, "stage2")


def build_alexnet_small_silu_poly_graph(inputs: dict) -> TensorTerm:
    """Full AlexNetSmall graph to logits (same as ``depth="full"``)."""
    return build_alexnet_small_silu_poly_graph_to_depth(inputs, "full")
