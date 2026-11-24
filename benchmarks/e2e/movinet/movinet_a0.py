"""MoViNet-A0 (stream) as a TensorTerm program.

The architecture is the public one from the TensorFlow Model Garden
(``official/projects/movinet``, config ``movinet_a0_stream``): causal,
(2+1)D convolutions, ``2plus3d`` squeeze-excitation, hard-swish, and a
hard-sigmoid gate. Tensors are ``[C, T, H, W]`` with frames on ``T``.

- ``A0``: the official widths, kernels and strides
- ``init_weights``: random weights, batch norm folded into each conv
- ``reference``: the forward pass in PyTorch
- ``tensor_ir``: the same forward pass as a TensorTerm

Each block is::

    expand 1x1x1 -> depthwise 1xkxk (stride) -> depthwise ktx1x1 (causal)
        -> squeeze-excitation gate -> project 1x1x1, plus the skip

The activations are exact functions (``hard_swish`` and a clipped linear
gate), which the toy backend evaluates as they are; they are not polynomial
approximations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from frontends.tensor import TensorTerm
from frontends.tensor_args import tf_same_padding

Weights = Dict[str, Tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Layer:
    """One bottleneck layer: the expanded width, and the kernel and stride of
    its depthwise conv as ``(T, H, W)``."""

    expand: int
    kernel: Tuple[int, int, int]
    stride: Tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True)
class Block:
    filters: int
    layers: Tuple[Layer, ...]


@dataclass(frozen=True)
class Config:
    stem: int
    blocks: Tuple[Block, ...]
    head: int
    hidden: int
    classes: int


_DOWN = (1, 2, 2)

A0 = Config(
    stem=8,
    blocks=(
        Block(8, (Layer(24, (1, 5, 5), _DOWN),)),
        Block(
            32,
            (
                Layer(80, (3, 3, 3), _DOWN),
                Layer(80, (3, 3, 3)),
                Layer(80, (3, 3, 3)),
            ),
        ),
        Block(
            56,
            (
                Layer(184, (5, 3, 3), _DOWN),
                Layer(112, (3, 3, 3)),
                Layer(184, (3, 3, 3)),
            ),
        ),
        Block(
            56,
            (
                Layer(184, (5, 3, 3)),
                Layer(184, (3, 3, 3)),
                Layer(184, (3, 3, 3)),
                Layer(184, (3, 3, 3)),
            ),
        ),
        Block(
            104,
            (
                Layer(384, (5, 3, 3), _DOWN),
                Layer(280, (1, 5, 5)),
                Layer(280, (1, 5, 5)),
                Layer(344, (1, 5, 5)),
            ),
        ),
    ),
    head=480,
    hidden=2048,
    classes=600,
)

# The same block structure at a size a test can compile: a downsampling layer
# without a temporal conv, one with, and a layer with an identity skip.
TINY = Config(
    stem=4,
    blocks=(
        Block(4, (Layer(8, (1, 3, 3), _DOWN),)),
        Block(8, (Layer(12, (3, 3, 3), _DOWN), Layer(12, (3, 3, 3)))),
    ),
    head=12,
    hidden=16,
    classes=4,
)


def se_filters(expand: int) -> int:
    """The squeeze width of a ``2plus3d`` squeeze-excitation: half the
    expanded width, rounded to a multiple of 8."""
    return max(8, int(expand * 0.5 + 4) // 8 * 8)


def layers(config: Config):
    """Every bottleneck layer as ``(name prefix, block, layer)``, in order."""
    for b, block in enumerate(config.blocks):
        for i, layer in enumerate(block.layers):
            yield f"b{b}l{i}/", block, layer


def sites(config: Config):
    """Every conv of the network as ``(name, weight shape)``, in order.

    A weight is ``[C_out, C_in, K_t, K_h, K_w]``, or ``[C, 1, K_t, K_h, K_w]``
    for a depthwise conv (PyTorch's layout).
    """
    out = [("stem", (config.stem, 3, 1, 3, 3))]
    channels = config.stem
    for p, block, layer in layers(config):
        kt, kh, kw = layer.kernel
        squeeze = se_filters(layer.expand)
        out.append((p + "expand", (layer.expand, channels, 1, 1, 1)))
        out.append((p + "spatial", (layer.expand, 1, 1, kh, kw)))
        if kt > 1:
            out.append((p + "temporal", (layer.expand, 1, kt, 1, 1)))
        # the squeeze reads the cumulative average, then the frame's mean
        out.append((p + "se_reduce", (squeeze, 2 * layer.expand, 1, 1, 1)))
        out.append((p + "se_expand", (layer.expand, squeeze, 1, 1, 1)))
        out.append((p + "project", (block.filters, layer.expand, 1, 1, 1)))
        if channels != block.filters or layer.stride != (1, 1, 1):
            out.append((p + "skip", (block.filters, channels, 1, 1, 1)))
        channels = block.filters
    out.append(("head", (config.head, channels, 1, 1, 1)))
    out.append(("hidden", (config.hidden, config.head, 1, 1, 1)))
    out.append(("classifier", (config.classes, config.hidden, 1, 1, 1)))
    return out


def init_weights(config: Config, seed: int = 0) -> Weights:
    """Random ``(weight, bias)`` per conv, scaled to keep activations O(1)."""
    rng = np.random.default_rng(seed)
    weights = {}
    for name, shape in sites(config):
        fan_in = int(np.prod(shape[1:]))
        weights[name] = (
            rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=shape),
            rng.normal(0.0, 0.1, size=shape[0]),
        )
    return weights


def from_official(variables, config: Config = A0, epsilon: float = 1e-3) -> Weights:
    """The weights of the Model Garden's ``MovinetClassifier``, folded.

    Args:
        variables: the Keras model's variables by name (``{v.name: v.numpy()}``)
        epsilon: the model's batch norm epsilon

    Batch norm and the block's rezero scale go into the conv before them.
    """

    def one(prefix, suffix):
        names = [n for n in variables if n.startswith(prefix) and n.endswith(suffix)]
        if len(names) != 1:
            raise KeyError(f"{prefix}*{suffix}: {names}")
        return np.asarray(variables[names[0]], dtype=np.float64)

    def has(prefix, suffix):
        return any(n.startswith(prefix) and n.endswith(suffix) for n in variables)

    def site(prefix, scale=1.0):
        if has(prefix, "depthwise_kernel:0"):
            # [K_a, K_b, C, 1], over (H, W) or over (T, 1)
            kernel = np.transpose(one(prefix, "depthwise_kernel:0"), (2, 3, 0, 1))
            if "temporal" in prefix:
                weight = kernel.reshape(kernel.shape[0], 1, kernel.shape[2], 1, 1)
            else:
                weight = kernel[:, :, None]
        else:
            # [K_h, K_w, C_in, C_out]
            weight = np.transpose(one(prefix, "kernel:0"), (3, 2, 0, 1))[:, :, None]
        if has(prefix, "gamma:0"):
            gain = one(prefix, "gamma:0") / np.sqrt(
                one(prefix, "moving_variance:0") + epsilon
            )
            shift = one(prefix, "beta:0") - one(prefix, "moving_mean:0") * gain
        else:
            gain, shift = np.ones(weight.shape[0]), one(prefix, "bias:0")
        gain, shift = gain * scale, shift * scale
        return weight * gain[:, None, None, None, None], shift

    parts = {
        "expand": "expansion/",
        "spatial": "feature/conv2d/",
        "temporal": "feature/conv2d_temporal/",
        "se_reduce": "se/se_reduce/",
        "se_expand": "se/se_expand/",
        "project": "projection/",
        "skip": "skip/",
    }
    top = {
        "stem": "stem/",
        "head": "head/",
        "hidden": "classifier_head/head/",
        "classifier": "classifier_head/classifier/",
    }
    weights = {}
    for name, shape in sites(config):
        if name in top:
            weights[name] = site(top[name])
        else:
            layer, part = name.split("/")
            block = f"block{layer[1 : layer.index('l')]}_layer{layer[layer.index('l') + 1 :]}"
            scale = float(variables[f"{block}/scale:0"]) if part == "project" else 1.0
            weights[name] = site(f"{block}/bneck/{parts[part]}", scale)
        if weights[name][0].shape != tuple(shape):
            raise ValueError(f"{name}: {weights[name][0].shape} is not {shape}")
    return weights


def _spatial_padding(height: int, width: int, layer_kernel, stride):
    """TensorFlow ``SAME`` on H and W, nothing in time."""
    return (
        (0, 0),
        tf_same_padding(height, layer_kernel[1], stride[1]),
        tf_same_padding(width, layer_kernel[2], stride[2]),
    )


def _out_extent(extent: int, stride: int) -> int:
    return -(-extent // stride)


def pool_correction(height: int, width: int) -> np.ndarray:
    """``9 / count`` for a 3x3, stride 2, ``SAME`` average pool, as
    ``[1, 1, H_out, W_out]``: TensorFlow averages over the cells inside the
    image, so a window that hangs over the edge divides by fewer."""

    def counts(extent):
        before, _ = tf_same_padding(extent, 3, 2)
        starts = np.arange(_out_extent(extent, 2)) * 2 - before
        return np.minimum(starts + 3, extent) - np.maximum(starts, 0)

    return (9.0 / np.outer(counts(height), counts(width)))[None, None]


# -----------------------------------------------------------------------------
# PyTorch reference
# -----------------------------------------------------------------------------


def reference(config: Config, weights: Weights, x: np.ndarray) -> np.ndarray:
    """The logits ``[classes]`` of a clip ``x`` of shape ``[3, T, H, W]``."""
    import torch
    import torch.nn.functional as F

    def conv(h, name, stride=(1, 1, 1), causal=False, depthwise=False):
        w, b = (
            torch.from_numpy(np.asarray(a, dtype=np.float64)) for a in weights[name]
        )
        kt, kh, kw = w.shape[2:]
        pads = (
            (kt - 1, 0) if causal else (0, 0),
            tf_same_padding(h.shape[2], kh, stride[1]),
            tf_same_padding(h.shape[3], kw, stride[2]),
        )
        h = F.pad(h[None], (*pads[2], *pads[1], *pads[0]))
        groups = h.shape[1] if depthwise else 1
        return F.conv3d(h, w, b, stride=stride, groups=groups)[0]

    def hard_swish(h):
        return h * torch.clamp(h + 3, 0, 6) / 6

    def hard_sigmoid(h):
        return torch.clamp(h + 3, 0, 6) / 6

    h = torch.from_numpy(np.asarray(x, dtype=np.float64))
    h = hard_swish(conv(h, "stem", _DOWN))
    for p, block, layer in layers(config):
        skip = h
        h = hard_swish(conv(h, p + "expand"))
        h = hard_swish(conv(h, p + "spatial", layer.stride, depthwise=True))
        if layer.kernel[0] > 1:
            h = hard_swish(conv(h, p + "temporal", causal=True, depthwise=True))

        frame_mean = h.mean((2, 3), keepdim=True)
        frames = frame_mean.shape[1]
        count = torch.arange(1, frames + 1).reshape(1, frames, 1, 1)
        cumulative = torch.cumsum(frame_mean, 1) / count
        gate = torch.cat([cumulative, frame_mean], 0)
        gate = hard_swish(conv(gate, p + "se_reduce"))
        gate = hard_sigmoid(conv(gate, p + "se_expand"))
        h = conv(h * gate, p + "project")

        if p + "skip" in weights:
            if layer.stride != (1, 1, 1):
                pads = (
                    *tf_same_padding(skip.shape[3], 3, 2),
                    *tf_same_padding(skip.shape[2], 3, 2),
                )
                total = F.avg_pool2d(F.pad(skip, pads), 3, 2, divisor_override=1)
                cells = F.avg_pool2d(
                    F.pad(torch.ones_like(skip), pads), 3, 2, divisor_override=1
                )
                skip = total / cells
            skip = conv(skip, p + "skip")
        h = h + skip

    h = hard_swish(conv(h, "head"))
    h = h.mean((1, 2, 3), keepdim=True)
    h = hard_swish(conv(h, "hidden"))
    return conv(h, "classifier").reshape(-1).numpy()


# -----------------------------------------------------------------------------
# TensorTerm program
# -----------------------------------------------------------------------------


def tensor_ir(config: Config, weights: Weights, x: np.ndarray):
    """The network over a secret clip ``x`` of shape ``[3, T, H, W]``.

    Returns:
        (term, inputs): the logits as ``[classes, 1, 1, 1]`` and the values of
        every tensor the term names
    """
    _, frames, height, width = x.shape
    inputs = {"x": np.asarray(x, dtype=np.float64)}

    def public(name, value):
        inputs[name] = np.asarray(value, dtype=np.float64)
        return TensorTerm.Tensor(name, list(inputs[name].shape), False)

    def bias(name):
        return public(name + "/b", weights[name][1].reshape(-1, 1, 1, 1))

    def conv(h, name, stride=1, padding="same"):
        w = public(name + "/w", weights[name][0])
        return TensorTerm.conv3d(h, w, stride, padding) + bias(name)

    def depthwise(h, name, stride, padding):
        # [C, 1, K_t, K_h, K_w] -> [1, K_t, K_h, K_w, C]
        w = public(name + "/w", np.transpose(weights[name][0], (1, 2, 3, 4, 0)))
        return TensorTerm.depthwise_conv3d(h, w, stride, padding) + bias(name)

    # 1 / (t + 1): the divisor of the cumulative average at frame t
    frame_scale = public(
        "frame_scale", (1.0 / np.arange(1, frames + 1)).reshape(1, frames, 1, 1)
    )
    causal_all = ((frames - 1, 0), (0, 0), (0, 0))

    def gate(h, p, channels):
        """The squeeze-excitation gate of ``h``, as ``[C, T, 1, 1]``."""
        frame_mean = h.mean((2, 3))
        w, _ = weights[p + "se_reduce"]
        # The squeeze is linear in [cumulative average, frame mean]. A causal
        # conv over every frame with the same taps is the cumulative sum.
        w_cumulative = public(
            p + "se_reduce/w_cumulative", np.repeat(w[:, :channels], frames, axis=2)
        )
        w_frame = public(p + "se_reduce/w_frame", w[:, channels:])
        squeezed = (
            TensorTerm.conv3d(frame_mean, w_cumulative, 1, causal_all) * frame_scale
            + TensorTerm.conv3d(frame_mean, w_frame, 1, "same")
            + bias(p + "se_reduce")
        ).hard_swish()
        half = public(p + "half", np.full((channels, frames, 1, 1), 0.5))
        return conv(squeezed, p + "se_expand").hard_sigmoid_centered() + half

    def skip_path(h, p, layer, height, width):
        if p + "skip" not in weights:
            return h
        if layer.stride == (1, 1, 1):
            return conv(h, p + "skip")
        # An average pool then a 1x1 conv is one 3x3 conv, scaled where the
        # window hangs over the edge.
        w, _ = weights[p + "skip"]
        w = public(p + "skip/w", np.tile(w, (1, 1, 1, 3, 3)) / 9.0)
        pooled = TensorTerm.conv3d(
            h, w, layer.stride, _spatial_padding(height, width, (1, 3, 3), layer.stride)
        )
        correction = public(p + "skip/correction", pool_correction(height, width))
        return pooled * correction + bias(p + "skip")

    h = TensorTerm.Tensor("x", [3, frames, height, width], True)
    h = conv(h, "stem", _DOWN, _spatial_padding(height, width, (1, 3, 3), _DOWN))
    h = h.hard_swish()
    height, width = _out_extent(height, 2), _out_extent(width, 2)

    for p, block, layer in layers(config):
        skip = skip_path(h, p, layer, height, width)

        h = conv(h, p + "expand").hard_swish()
        h = depthwise(
            h,
            p + "spatial",
            layer.stride,
            _spatial_padding(height, width, layer.kernel, layer.stride),
        ).hard_swish()
        height = _out_extent(height, layer.stride[1])
        width = _out_extent(width, layer.stride[2])
        if layer.kernel[0] > 1:
            causal = ((layer.kernel[0] - 1, 0), (0, 0), (0, 0))
            h = depthwise(h, p + "temporal", 1, causal).hard_swish()

        h = h * gate(h, p, layer.expand)
        h = conv(h, p + "project") + skip

    h = conv(h, "head").hard_swish()
    h = h.mean((1, 2, 3))
    h = conv(h, "hidden").hard_swish()
    return conv(h, "classifier"), inputs
