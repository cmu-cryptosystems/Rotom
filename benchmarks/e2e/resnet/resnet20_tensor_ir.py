"""TensorTerm IR for CIFAR ResNet-20 without activation functions.

Matches ``resnet_model.resnet20`` topology (Option-A shortcut): conv/BN only—no SiLU.
Batch norm is plaintext ``x * scale + shift``. Global pooling uses ``sum`` over H/W;
``fc`` weights are scaled by ``1/64`` so ``sum`` matches PyTorch ``avg_pool2d`` + ``Linear``.

Shortcuts use a fixed 1×1 stride-2 conv (plaintext) equivalent to Option A padding
(no checkpoint weights for those tensors).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from frontends.tensor import TensorTerm


def option_a_shortcut_weights(in_ch: int, out_ch: int, planes: int) -> np.ndarray:
    pad = planes // 4
    w = np.zeros((out_ch, in_ch, 1, 1), dtype=np.float64)
    for i in range(in_ch):
        w[pad + i, i, 0, 0] = 1.0
    return w


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
    """Output (H, W) for 3×3 conv with symmetric pad 1 (PyTorch-style), matching tensor_evaluator."""
    p = 1
    k = 3
    h_pad = h + 2 * p
    w_pad = w + 2 * p
    return (h_pad - k) // stride + 1, (w_pad - k) // stride + 1


def _bn_affine_hw(
    x: TensorTerm, prefix: str, num_ch: int, h: int, w: int, inputs: dict
) -> TensorTerm:
    """Affine BN as ``x * scale + shift`` with full ``[C,H,W]`` plaintext tensors.

    Layout assignment does not support broadcast ``[C,H,W] * [C,1,1]``; we tile
    the per-channel scale/shift to match the feature map.
    """
    sk = f"{prefix}_scale_hw_{h}_{w}"
    shk = f"{prefix}_shift_hw_{h}_{w}"
    if sk not in inputs:
        base = inputs[f"{prefix}_scale"]
        inputs[sk] = np.broadcast_to(base, (num_ch, h, w)).astype(np.float64).copy()
    if shk not in inputs:
        base = inputs[f"{prefix}_shift"]
        inputs[shk] = np.broadcast_to(base, (num_ch, h, w)).astype(np.float64).copy()
    sc = TensorTerm.Tensor(sk, [num_ch, h, w], False)
    shift = TensorTerm.Tensor(shk, [num_ch, h, w], False)
    return x * sc + shift


def _conv_same(x: TensorTerm, weight_key: str, inputs: dict, stride: int) -> TensorTerm:
    w = _w_tensor(inputs, weight_key)
    return TensorTerm.conv2d(x, w, stride, "same")


def _basic_block_no_act(
    x: TensorTerm,
    block_id: str,
    in_ch: int,
    out_ch: int,
    stride: int,
    inputs: dict,
    h: int,
    w: int,
) -> tuple[TensorTerm, int, int]:
    out = _conv_same(x, f"{block_id}_conv1_w", inputs, stride)
    h1, w1 = _spatial_hw_after_conv3(h, w, stride)
    out = _bn_affine_hw(out, f"{block_id}_bn1", out_ch, h1, w1, inputs)
    out = _conv_same(out, f"{block_id}_conv2_w", inputs, 1)
    out = _bn_affine_hw(out, f"{block_id}_bn2", out_ch, h1, w1, inputs)

    if stride != 1 or in_ch != out_ch:
        sk = f"{block_id}_shortcut_w"
        inputs[sk] = option_a_shortcut_weights(in_ch, out_ch, out_ch)
        sw = _w_tensor(inputs, sk)
        shortcut = TensorTerm.conv2d(x, sw, 2, "same")
    else:
        shortcut = x
    return out + shortcut, h1, w1


def _fill_basic_block(
    inputs: dict, block: nn.Module, block_id: str, out_ch: int
) -> None:
    _fill_conv_weights(inputs, f"{block_id}_conv1_w", block.conv1)
    _fill_bn_affine(inputs, f"{block_id}_bn1", block.bn1)
    _fill_conv_weights(inputs, f"{block_id}_conv2_w", block.conv2)
    _fill_bn_affine(inputs, f"{block_id}_bn2", block.bn2)


def populate_resnet20_no_activation_inputs(model: nn.Module, inputs: dict) -> None:
    """Fill ``inputs`` with numpy weights for :func:`build_resnet20_no_activation_graph`."""
    inputs.clear()
    _fill_conv_weights(inputs, "conv1_w", model.conv1)
    _fill_bn_affine(inputs, "bn1", model.bn1)

    W = model.linear.weight.detach().cpu().numpy()
    b = model.linear.bias.detach().cpu().numpy()
    inputs["fc"] = (W.T / 64.0).astype(np.float64)
    inputs["fc_b"] = b.reshape(1, 10).astype(np.float64)

    for i in range(3):
        bid = f"l1_{i}"
        _fill_basic_block(inputs, model.layer1[i], bid, 16)

    _fill_basic_block(inputs, model.layer2[0], "l2_0", 32)
    inputs["l2_0_shortcut_w"] = option_a_shortcut_weights(16, 32, 32)
    for i in range(1, 3):
        _fill_basic_block(inputs, model.layer2[i], f"l2_{i}", 32)

    _fill_basic_block(inputs, model.layer3[0], "l3_0", 64)
    inputs["l3_0_shortcut_w"] = option_a_shortcut_weights(32, 64, 64)
    for i in range(1, 3):
        _fill_basic_block(inputs, model.layer3[i], f"l3_{i}", 64)


def build_resnet20_no_activation_graph(inputs: dict) -> TensorTerm:
    """Build TensorTerm graph; ``inputs`` must already be populated."""
    h, w = 32, 32
    x = TensorTerm.Tensor("input", [3, 32, 32], True)
    x = _conv_same(x, "conv1_w", inputs, 1)
    h, w = _spatial_hw_after_conv3(h, w, 1)
    x = _bn_affine_hw(x, "bn1", 16, h, w, inputs)

    for i in range(3):
        x, h, w = _basic_block_no_act(x, f"l1_{i}", 16, 16, 1, inputs, h, w)

    x, h, w = _basic_block_no_act(x, "l2_0", 16, 32, 2, inputs, h, w)
    for i in range(1, 3):
        x, h, w = _basic_block_no_act(x, f"l2_{i}", 32, 32, 1, inputs, h, w)

    x, h, w = _basic_block_no_act(x, "l3_0", 32, 64, 2, inputs, h, w)
    for i in range(1, 3):
        x, h, w = _basic_block_no_act(x, f"l3_{i}", 64, 64, 1, inputs, h, w)

    x = x.sum(1)
    x = x.sum(1)
    x = x.reshape(0, {0: 1, 1: 64})
    fc = _w_tensor(inputs, "fc")
    fb = TensorTerm.Tensor("fc_b", [1, 10], False)
    return x @ fc + fb


def _silu_poly(x: TensorTerm) -> TensorTerm:
    # Keep SiLU as POLY_CALL so TensorEvaluator and Toy backend share
    # the same clipped silu_approx implementation.
    return x.poly_call("silu", -20.0, 20.0)


def _basic_block_silu_poly(
    x: TensorTerm,
    block_id: str,
    in_ch: int,
    out_ch: int,
    stride: int,
    inputs: dict,
    h: int,
    w: int,
) -> tuple[TensorTerm, int, int]:
    """BasicBlock with affine BN and ``silu`` polynomial approx (matches PyTorch order)."""
    out = _conv_same(x, f"{block_id}_conv1_w", inputs, stride)
    h1, w1 = _spatial_hw_after_conv3(h, w, stride)
    out = _bn_affine_hw(out, f"{block_id}_bn1", out_ch, h1, w1, inputs)
    out = _silu_poly(out)
    out = _conv_same(out, f"{block_id}_conv2_w", inputs, 1)
    out = _bn_affine_hw(out, f"{block_id}_bn2", out_ch, h1, w1, inputs)

    if stride != 1 or in_ch != out_ch:
        sk = f"{block_id}_shortcut_w"
        inputs[sk] = option_a_shortcut_weights(in_ch, out_ch, out_ch)
        sw = _w_tensor(inputs, sk)
        shortcut = TensorTerm.conv2d(x, sw, 2, "same")
    else:
        shortcut = x
    out = out + shortcut
    out = _silu_poly(out)
    return out, h1, w1


def build_resnet20_silu_poly_graph_through_layer1(inputs: dict) -> TensorTerm:
    """Stem + layer1 (three BasicBlocks) with SiLU poly; all ops stay on 32×32."""
    return build_resnet20_silu_poly_graph_to_depth(inputs, "l1")


def build_resnet20_silu_poly_l2_0_block_graph(inputs: dict) -> TensorTerm:
    """Only the layer2.0 stride-2 BasicBlock (SiLU poly).

    Secret activations must be supplied as ``inputs["l2_0_block_in"]`` with shape
    ``[16, 32, 32]`` (e.g. by evaluating :func:`build_resnet20_silu_poly_graph_through_layer1`
    on the same ``inputs``). Weights for ``l2_0`` are read from ``inputs`` like the
    full-graph builders.
    """
    h, w = 32, 32
    x = TensorTerm.Tensor("l2_0_block_in", [16, 32, 32], True)
    x, _h, _w = _basic_block_silu_poly(x, "l2_0", 16, 32, 2, inputs, h, w)
    return x


def build_resnet20_silu_poly_l1_block_graph(inputs: dict, index: int) -> TensorTerm:
    """Single layer1 BasicBlock ``l1_{index}`` (stride 1, 16→16, 32×32).

    Secret activations: ``inputs[f"l1_{index}_block_in"]`` with shape ``[16, 32, 32]``.
    ``index`` must be ``0``, ``1``, or ``2``.
    """
    if index not in (0, 1, 2):
        raise ValueError(f"index must be 0, 1, or 2, got {index!r}")
    h, w = 32, 32
    key = f"l1_{index}_block_in"
    x = TensorTerm.Tensor(key, [16, 32, 32], True)
    x, _h, _w = _basic_block_silu_poly(x, f"l1_{index}", 16, 16, 1, inputs, h, w)
    return x


def build_resnet20_silu_poly_layer1_only_graph(inputs: dict) -> TensorTerm:
    """All three layer1 BasicBlocks only (no stem): ``l1_0`` … ``l1_2``.

    Secret activations: ``inputs["layer1_only_in"]`` with shape ``[16, 32, 32]``.
    """
    h, w = 32, 32
    x = TensorTerm.Tensor("layer1_only_in", [16, 32, 32], True)
    for i in range(3):
        x, h, w = _basic_block_silu_poly(x, f"l1_{i}", 16, 16, 1, inputs, h, w)
    return x


SiluPolyDepth = Literal["stem", "l1", "l2_0", "l2", "l3", "full"]


def build_resnet20_silu_poly_graph_to_depth(
    inputs: dict, depth: SiluPolyDepth
) -> TensorTerm:
    """SiLU-poly ResNet-20 prefix: stop after ``stem``, ``l1``, ``l2_0``, ``l2``, ``l3``, or ``full`` (FC)."""
    h, w = 32, 32
    x = TensorTerm.Tensor("input", [3, 32, 32], True)
    x = _conv_same(x, "conv1_w", inputs, 1)
    h, w = _spatial_hw_after_conv3(h, w, 1)
    x = _bn_affine_hw(x, "bn1", 16, h, w, inputs)
    x = _silu_poly(x)
    if depth == "stem":
        return x

    for i in range(3):
        x, h, w = _basic_block_silu_poly(x, f"l1_{i}", 16, 16, 1, inputs, h, w)
    if depth == "l1":
        return x

    x, h, w = _basic_block_silu_poly(x, "l2_0", 16, 32, 2, inputs, h, w)
    if depth == "l2_0":
        return x

    for i in range(1, 3):
        x, h, w = _basic_block_silu_poly(x, f"l2_{i}", 32, 32, 1, inputs, h, w)
    if depth == "l2":
        return x

    x, h, w = _basic_block_silu_poly(x, "l3_0", 32, 64, 2, inputs, h, w)
    for i in range(1, 3):
        x, h, w = _basic_block_silu_poly(x, f"l3_{i}", 64, 64, 1, inputs, h, w)
    if depth == "l3":
        return x

    x = x.sum(1)
    x = x.sum(1)
    x = x.reshape(0, {0: 1, 1: 64})
    fc = _w_tensor(inputs, "fc")
    fb = TensorTerm.Tensor("fc_b", [1, 10], False)
    return x @ fc + fb


def build_resnet20_silu_poly_graph(inputs: dict) -> TensorTerm:
    """Full CIFAR ResNet-20 with ``poly_call('silu', ...)`` at each activation site."""
    return build_resnet20_silu_poly_graph_to_depth(inputs, "full")


@torch.no_grad()
def resnet20_forward_no_activation(
    model: nn.Module, x_bchw: torch.Tensor
) -> torch.Tensor:
    """Reference forward: same topology as the TensorTerm graph (no SiLU)."""
    x = model.conv1(x_bchw)
    x = model.bn1(x)
    for i in range(3):
        x = _basic_block_forward_no_act(model.layer1[i], x)
    x = _basic_block_forward_no_act(model.layer2[0], x)
    for i in range(1, 3):
        x = _basic_block_forward_no_act(model.layer2[i], x)
    x = _basic_block_forward_no_act(model.layer3[0], x)
    for i in range(1, 3):
        x = _basic_block_forward_no_act(model.layer3[i], x)
    x = F.avg_pool2d(x, x.size()[3])
    x = x.view(x.size(0), -1)
    return model.linear(x)


def _basic_block_forward_no_act(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = block.conv1(x)
    out = block.bn1(out)
    out = block.conv2(out)
    out = block.bn2(out)
    return out + block.shortcut(x)
