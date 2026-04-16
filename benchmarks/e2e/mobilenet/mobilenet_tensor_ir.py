"""TensorTerm IR for :class:`mobilenet_model.MobileNetSmall` (SiLU via ``poly_call('silu', ...)``).

Depth checkpoints (see :func:`build_mobilenet_small_silu_poly_graph_to_depth`):

- ``stem``: stem conv + affine BN + SiLU (``[16, 32, 32]``).
- ``stage2``: stem + first MB-style block (3×3 expand + 1×1 project), ``[16, 16, 16]``.
- ``stage3``: + second block, ``[24, 8, 8]``.
- ``full``: + third block, global sum-pool (scaled like avg pool) + linear logits.

Affine BN and conv helpers match :mod:`benchmarks.e2e.resnet.resnet20_tensor_ir`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch.nn as nn
from frontends.tensor import TensorTerm

from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    _bn_affine_hw,
    _conv_same,
    _fill_bn_affine,
    _fill_conv_weights,
    _silu_poly,
    _spatial_hw_after_conv3,
    _w_tensor,
)
from util.silu_polycall_eval import (
    DEFAULT_SILU_POLY_DEGREE,
    DEFAULT_SILU_POLY_NODES,
    SILU_POLY_DEGREE_KEY,
    SILU_POLY_NODES_KEY,
)


def populate_mobilenet_small_inputs(model: nn.Module, inputs: dict) -> None:
    """Populate numpy weights for :func:`build_mobilenet_small_silu_poly_graph_to_depth`."""
    inputs.clear()
    _fill_conv_weights(inputs, "conv_stem_w", model.conv_stem)
    _fill_bn_affine(inputs, "bn_stem", model.bn_stem)

    _fill_conv_weights(inputs, "conv_b1_exp_w", model.conv_b1_exp)
    _fill_bn_affine(inputs, "bn_b1_exp", model.bn_b1_exp)
    _fill_conv_weights(inputs, "conv_b1_pw_w", model.conv_b1_pw)
    _fill_bn_affine(inputs, "bn_b1_pw", model.bn_b1_pw)

    _fill_conv_weights(inputs, "conv_b2_exp_w", model.conv_b2_exp)
    _fill_bn_affine(inputs, "bn_b2_exp", model.bn_b2_exp)
    _fill_conv_weights(inputs, "conv_b2_pw_w", model.conv_b2_pw)
    _fill_bn_affine(inputs, "bn_b2_pw", model.bn_b2_pw)

    _fill_conv_weights(inputs, "conv_b3_exp_w", model.conv_b3_exp)
    _fill_bn_affine(inputs, "bn_b3_exp", model.bn_b3_exp)
    _fill_conv_weights(inputs, "conv_b3_pw_w", model.conv_b3_pw)
    _fill_bn_affine(inputs, "bn_b3_pw", model.bn_b3_pw)

    w = model.fc.weight.detach().cpu().numpy()
    b = model.fc.bias.detach().cpu().numpy()
    inputs["fc"] = (w.T / 64.0).astype(np.float64)
    inputs["fc_b"] = b.reshape(1, -1).astype(np.float64)

    inputs[SILU_POLY_DEGREE_KEY] = DEFAULT_SILU_POLY_DEGREE
    inputs[SILU_POLY_NODES_KEY] = DEFAULT_SILU_POLY_NODES


MobileNetSmallDepth = Literal["stem", "stage2", "stage3", "full"]


def build_mobilenet_small_silu_poly_graph_to_depth(
    inputs: dict, depth: MobileNetSmallDepth
) -> TensorTerm:
    """SiLU-poly MobileNetSmall prefix; stop after ``stem``, ``stage2``, ``stage3``, or ``full``."""
    h, w = 32, 32
    x = TensorTerm.Tensor("input", [3, 32, 32], True)

    x = _conv_same(x, "conv_stem_w", inputs, 1)
    h, w = _spatial_hw_after_conv3(h, w, 1)
    x = _bn_affine_hw(x, "bn_stem", 16, h, w, inputs)
    x = _silu_poly(x)
    if depth == "stem":
        return x

    x = _conv_same(x, "conv_b1_exp_w", inputs, 2)
    h, w = _spatial_hw_after_conv3(h, w, 2)
    x = _bn_affine_hw(x, "bn_b1_exp", 32, h, w, inputs)
    x = _silu_poly(x)
    x = _conv_same(x, "conv_b1_pw_w", inputs, 1)
    x = _bn_affine_hw(x, "bn_b1_pw", 16, h, w, inputs)
    x = _silu_poly(x)
    if depth == "stage2":
        return x

    x = _conv_same(x, "conv_b2_exp_w", inputs, 2)
    h, w = _spatial_hw_after_conv3(h, w, 2)
    x = _bn_affine_hw(x, "bn_b2_exp", 32, h, w, inputs)
    x = _silu_poly(x)
    x = _conv_same(x, "conv_b2_pw_w", inputs, 1)
    x = _bn_affine_hw(x, "bn_b2_pw", 24, h, w, inputs)
    x = _silu_poly(x)
    if depth == "stage3":
        return x

    x = _conv_same(x, "conv_b3_exp_w", inputs, 1)
    h, w = _spatial_hw_after_conv3(h, w, 1)
    x = _bn_affine_hw(x, "bn_b3_exp", 48, h, w, inputs)
    x = _silu_poly(x)
    x = _conv_same(x, "conv_b3_pw_w", inputs, 1)
    x = _bn_affine_hw(x, "bn_b3_pw", 32, h, w, inputs)
    x = _silu_poly(x)

    x = x.sum(1)
    x = x.sum(1)
    x = x.reshape(0, {0: 1, 1: 32})
    fcb = np.asarray(inputs["fc_b"])
    x = x @ _w_tensor(inputs, "fc") + TensorTerm.Tensor("fc_b", list(fcb.shape), False)
    return x


def build_mobilenet_small_silu_poly_graph(inputs: dict) -> TensorTerm:
    """Full graph to logits (same as ``depth=\"full\"``)."""
    return build_mobilenet_small_silu_poly_graph_to_depth(inputs, "full")


def build_mobilenet_small_linear_head_graph(inputs: dict) -> TensorTerm:
    """Classifier only: ``feat @ fc + fc_b`` (same ``fc`` / ``fc_b`` as :func:`populate_mobilenet_small_inputs`).

    ``feat`` is a secret rank-2 tensor ``[1, 32]`` matching pooled features before ``nn.Linear(32, 10)``.
    The caller must set ``inputs[\"feat\"]`` (and populate weights via ``populate_mobilenet_small_inputs``).
    """
    fcb = np.asarray(inputs["fc_b"])
    x = TensorTerm.Tensor("feat", [1, 32], True)
    return x @ _w_tensor(inputs, "fc") + TensorTerm.Tensor(
        "fc_b", list(fcb.shape), False
    )
