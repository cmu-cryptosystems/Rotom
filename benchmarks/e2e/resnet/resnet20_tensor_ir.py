"""TensorTerm IR for CIFAR ResNet-20 (SiLU via ``poly_call('silu', ...)``).

``resnet_model.resnet20`` uses ``nn.SiLU``; by default ``tensor_ir.eval`` and Toy both call
``util.silu_polycall_eval.eval_silu_polycall`` with the same ``inputs`` dict, bounds from each
``PolyCall("silu", lo, hi)``, and poly degree/node count pinned by
:func:`populate_resnet20_inputs`. Each SiLU site can have bounds of its own
(``RESNET20_DACAPO_SILU_BOUNDS`` for the DaCapo checkpoint). Set
``inputs["__rotom_silu_eval_mode"] = "exact"`` to match
PyTorch's exact SiLU. Batch norm
is plaintext ``x * scale + shift``. Global pooling
uses ``sum`` over H/W; ``fc`` weights are scaled by ``1/64`` so ``sum`` matches PyTorch
``avg_pool2d`` + ``Linear``.

Shortcuts use a fixed 1×1 stride-2 conv (plaintext) equivalent to Option A padding
(no checkpoint weights for those tensors).
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import torch.nn as nn
from benchmarks.e2e.resnet.resnet_data import CKPT_FILE, load_checkpoint
from benchmarks.e2e.resnet.resnet_model import resnet20
from frontends.tensor import TensorTerm
from util.silu_polycall_eval import (
    DEFAULT_SILU_POLY_DEGREE,
    DEFAULT_SILU_POLY_NODES,
    SILU_POLY_DEGREE_KEY,
    SILU_POLY_NODES_KEY,
)

RESNET20_SILU_BOUNDS_KEY = "__rotom_resnet20_silu_bounds"

# One fit for every SiLU site, when the inputs name no bounds of their own.
GLOBAL_SILU_BOUNDS = (-20.0, 20.0)

# The range of each SiLU's input for the DaCapo ``resnet20.silu.model``
# checkpoint: its minimum and maximum over the 50,000 CIFAR-10 training images
# (``silu_poly_calibration.calibrate_silu_input_bounds``, ``global_min`` and
# ``global_max``), rounded outward. Sites are named as in
# ``silu_poly_calibration.resnet20_silu_site_names``.
#
# A polynomial fitted to one site's range is far closer to SiLU there than one
# fitted to [-20, 20]: with degree 15 the checkpoint keeps 92.1% top-1 on the
# CIFAR-10 test set (92.3% with exact SiLU), where one degree-11 fit on
# [-20, 20] gives 64.5%.
RESNET20_DACAPO_SILU_BOUNDS: dict[str, tuple[float, float]] = {
    "stem_conv1_bn": (-11.15, 11.46),
    "layer1_block0_post_conv1_bn": (-20.68, 10.62),
    "layer1_block0_post_add": (-15.09, 20.75),
    "layer1_block1_post_conv1_bn": (-12.48, 7.31),
    "layer1_block1_post_add": (-9.57, 16.61),
    "layer1_block2_post_conv1_bn": (-15.44, 8.93),
    "layer1_block2_post_add": (-8.84, 11.98),
    "layer2_block0_post_conv1_bn": (-8.98, 7.32),
    "layer2_block0_post_add": (-8.36, 13.63),
    "layer2_block1_post_conv1_bn": (-8.83, 4.94),
    "layer2_block1_post_add": (-7.86, 12.87),
    "layer2_block2_post_conv1_bn": (-8.47, 5.70),
    "layer2_block2_post_add": (-6.72, 14.60),
    "layer3_block0_post_conv1_bn": (-8.04, 7.26),
    "layer3_block0_post_add": (-9.49, 17.49),
    "layer3_block1_post_conv1_bn": (-8.02, 6.77),
    "layer3_block1_post_add": (-11.97, 20.75),
    "layer3_block2_post_conv1_bn": (-10.30, 9.12),
    "layer3_block2_post_add": (-25.91, 34.07),
}

# Degree 15 has the multiplicative depth of degree 11 (four levels) and fits
# the wider of these ranges; degree 11 on them gives 91.1%.
RESNET20_DACAPO_SILU_DEGREE = 15


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


def _fill_basic_block(
    inputs: dict, block: nn.Module, block_id: str, out_ch: int
) -> None:
    _fill_conv_weights(inputs, f"{block_id}_conv1_w", block.conv1)
    _fill_bn_affine(inputs, f"{block_id}_bn1", block.bn1)
    _fill_conv_weights(inputs, f"{block_id}_conv2_w", block.conv2)
    _fill_bn_affine(inputs, f"{block_id}_bn2", block.bn2)


def populate_resnet20_inputs(
    model: nn.Module,
    inputs: dict,
    silu_bounds: Mapping[str, tuple[float, float]] | None = None,
    silu_degree: int = DEFAULT_SILU_POLY_DEGREE,
) -> None:
    """Fill ``inputs`` with numpy weights for the ResNet-20 TensorTerm builders.

    Also sets ``SILU_POLY_DEGREE_KEY`` / ``SILU_POLY_NODES_KEY`` so ``tensor_ir.eval`` and
    Toy use the same SiLU polynomial fit (see ``util.silu_polycall_eval``).

    Args:
        model: a ``resnet_model.resnet20``
        inputs: cleared, then filled
        silu_bounds: the fit range of each SiLU site by name; every site uses
            ``GLOBAL_SILU_BOUNDS`` without it. The ranges belong to the weights:
            ``RESNET20_DACAPO_SILU_BOUNDS`` is for the DaCapo checkpoint.
        silu_degree: the degree of the SiLU polynomial
    """
    inputs.clear()
    if silu_bounds is not None:
        inputs[RESNET20_SILU_BOUNDS_KEY] = dict(silu_bounds)
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

    inputs[SILU_POLY_DEGREE_KEY] = int(silu_degree)
    inputs[SILU_POLY_NODES_KEY] = DEFAULT_SILU_POLY_NODES


def populate_dacapo_resnet20_inputs(
    inputs: dict, checkpoint_path: str = CKPT_FILE
) -> nn.Module:
    """Fill ``inputs`` from the DaCapo checkpoint, with its SiLU ranges and degree.

    Returns:
        The PyTorch model the weights were read from, in eval mode.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None or "state_dict" not in checkpoint:
        raise FileNotFoundError(f"Missing DaCapo checkpoint at {checkpoint_path}")
    model = resnet20(num_classes=10)
    # DaCapo saved the weights from a DataParallel model.
    model.load_state_dict(
        {
            key.removeprefix("module."): value
            for key, value in checkpoint["state_dict"].items()
        }
    )
    model.eval()
    populate_resnet20_inputs(
        model,
        inputs,
        silu_bounds=RESNET20_DACAPO_SILU_BOUNDS,
        silu_degree=RESNET20_DACAPO_SILU_DEGREE,
    )
    return model


def _silu_poly(x: TensorTerm, inputs: dict, site: str) -> TensorTerm:
    """``POLY_CALL("silu", lo, hi)`` on the range ``inputs`` gives the site."""
    bounds = inputs.get(RESNET20_SILU_BOUNDS_KEY)
    lo, hi = GLOBAL_SILU_BOUNDS if bounds is None else bounds[site]
    return x.poly_call("silu", float(lo), float(hi))


def _block_site(block_id: str, place: str) -> str:
    """The SiLU site of a block: ``("l2_0", "post_add")`` is ``layer2_block0_post_add``."""
    layer, block = block_id[1:].split("_")
    return f"layer{layer}_block{block}_{place}"


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
    out = _silu_poly(out, inputs, _block_site(block_id, "post_conv1_bn"))
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
    out = _silu_poly(out, inputs, _block_site(block_id, "post_add"))
    return out, h1, w1


def build_resnet20_silu_poly_graph_through_layer1(inputs: dict) -> TensorTerm:
    """Stem + layer1 (three BasicBlocks) with SiLU poly; all ops stay on 32×32."""
    return build_resnet20_silu_poly_graph_to_depth(inputs, "l1")


def build_resnet20_silu_poly_graph_through_layer2(inputs: dict) -> TensorTerm:
    """Stem + layer1 + layer2 (stride-2 block0 + two stride-1 blocks) with SiLU poly."""
    return build_resnet20_silu_poly_graph_to_depth(inputs, "l2")


def build_resnet20_silu_poly_graph_through_layer3(inputs: dict) -> TensorTerm:
    """Stem through layer3 (stride-2 l3.0 + two 64→64 blocks on 8×8) with SiLU poly.

    Stops before global pooling and the FC head (see ``depth == "l3"`` in
    :func:`build_resnet20_silu_poly_graph_to_depth`).
    """
    return build_resnet20_silu_poly_graph_to_depth(inputs, "l3")


def build_resnet20_stem_plus_layer1_convs_only(inputs: dict) -> TensorTerm:
    """Stem conv + all six layer1 3×3 convs (no BN, SiLU, or residual).

    Chains ``conv1`` then ``l1_i`` conv1/conv2 for ``i`` in ``0..2``. All stride-1
    ``same`` on ``32×32``; channel width stays 16 after the stem. Expects
    :func:`populate_resnet20_inputs` weight keys.
    """
    x = TensorTerm.Tensor("input", [3, 32, 32], True)
    x = _conv_same(x, "conv1_w", inputs, 1)
    for i in range(3):
        x = _conv_same(x, f"l1_{i}_conv1_w", inputs, 1)
        x = _conv_same(x, f"l1_{i}_conv2_w", inputs, 1)
    return x


def build_resnet20_stem_plus_l1_0_block_graph(inputs: dict) -> TensorTerm:
    """Stem + the first layer1 BasicBlock only (convs, affine BN, SiLU poly, residual).

    Matches the prefix of :func:`build_resnet20_silu_poly_graph_to_depth` through
    ``l1_0``. Expects ``populate_resnet20_inputs`` (or the same weight keys) and a
    secret ``"input"`` tensor ``[3, 32, 32]`` in ``inputs``.

    With default ``poly`` SiLU mode, ``tensor_ir.eval`` matches Toy on this graph.
    For eval vs PyTorch (exact SiLU), set ``inputs["__rotom_silu_eval_mode"] = "exact"``
    (see ``tests/e2e/resnet/test_resnet_silu.py``).
    """
    h, w = 32, 32
    x = TensorTerm.Tensor("input", [3, 32, 32], True)
    x = _conv_same(x, "conv1_w", inputs, 1)
    h, w = _spatial_hw_after_conv3(h, w, 1)
    x = _bn_affine_hw(x, "bn1", 16, h, w, inputs)
    x = _silu_poly(x, inputs, "stem_conv1_bn")
    x, h, w = _basic_block_silu_poly(x, "l1_0", 16, 16, 1, inputs, h, w)
    return x


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
    x = _silu_poly(x, inputs, "stem_conv1_bn")
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
