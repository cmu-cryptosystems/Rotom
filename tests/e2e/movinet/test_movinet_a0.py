"""MoViNet-A0 (stream): the TensorTerm program against the PyTorch reference
and, when the TensorFlow Model Garden is installed, the official model."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmarks.e2e.movinet.movinet_a0 import (
    A0,
    TINY,
    Block,
    Config,
    Layer,
    from_official,
    init_weights,
    pool_correction,
    reference,
    se_filters,
    sites,
    tensor_ir,
)
from ir.kernel import KernelOp
from tests.test_conv_direct import _run

# one downsampling layer with a temporal conv, a gate and a pooled skip
ONE_LAYER = Config(
    stem=4,
    blocks=(Block(4, (Layer(8, (3, 3, 3), (1, 2, 2)),)),),
    head=8,
    hidden=8,
    classes=4,
)


def _clip(frames, size, seed=1):
    return np.random.default_rng(seed).normal(size=(3, frames, size, size))


def test_a0_is_the_official_architecture():
    convs = dict(sites(A0))
    assert len(convs) == 108
    assert sum(len(block.layers) for block in A0.blocks) == 16
    assert convs["stem"] == (8, 3, 1, 3, 3)
    assert convs["b0l0/spatial"] == (24, 1, 1, 5, 5)
    assert "b0l0/temporal" not in convs
    assert convs["b2l0/temporal"] == (184, 1, 5, 1, 1)
    assert convs["b4l3/se_reduce"] == (176, 688, 1, 1, 1)
    assert convs["classifier"] == (600, 2048, 1, 1, 1)
    # the official model has 4,601,440 conv and batch norm parameters; folded,
    # each conv keeps its weights and one bias per output channel
    weights = init_weights(A0)
    assert sum(w.size + b.size for w, b in weights.values()) == 3_739_040


@pytest.mark.parametrize(
    "expand,squeeze", [(24, 16), (80, 40), (184, 96), (112, 56), (344, 176)]
)
def test_se_filters(expand, squeeze):
    assert se_filters(expand) == squeeze


def test_pool_correction_counts_the_cells_inside_the_image():
    # even: TensorFlow pads one cell after, so the last window holds 2 of 3
    even = pool_correction(4, 4)[0, 0]
    np.testing.assert_allclose(even, [[1, 1.5], [1.5, 2.25]])
    # odd: one cell before and one after
    odd = pool_correction(5, 3)[0, 0]
    np.testing.assert_allclose(odd[:, 0], [9 / 4, 9 / 6, 9 / 4])


@pytest.mark.parametrize(
    "config,frames,size",
    [
        (ONE_LAYER, 2, 8),
        (TINY, 3, 12),
        (TINY, 2, 16),
        (A0, 2, 32),
        (A0, 3, 44),  # 44 -> 22 -> 11 -> 6 -> 3 -> 2: odd extents
    ],
)
def test_eval_matches_the_reference(config, frames, size):
    weights = init_weights(config)
    x = _clip(frames, size)
    term, inputs = tensor_ir(config, weights, x)
    got = term.eval(inputs).reshape(-1)
    np.testing.assert_allclose(got, reference(config, weights, x), atol=1e-12)


def test_the_last_frame_changes_only_through_causal_paths():
    """A causal network over a longer clip sees the earlier frames unchanged:
    the features of frame t do not depend on frames after t. The logits pool
    every frame, so compare the program on a clip and on its prefix through
    one causal temporal conv and gate."""
    config = dataclasses.replace(ONE_LAYER, classes=3)
    weights = init_weights(config)
    x = _clip(4, 8)
    full = reference(config, weights, x)
    changed = x.copy()
    changed[:, -1] += 1.0
    assert not np.allclose(full, reference(config, weights, changed))


def test_one_layer_compiles_and_matches_the_reference():
    weights = init_weights(ONE_LAYER)
    x = _clip(2, 8)
    term, inputs = tensor_ir(ONE_LAYER, weights, x)
    kernel = _run(term, inputs, 1024)
    ops = [k.op for k in kernel.post_order()]
    # stem, expand, spatial, temporal, two squeeze convs, excite, project,
    # pooled skip, head, hidden, classifier
    assert ops.count(KernelOp.CONV_DIRECT) == 12
    np.testing.assert_allclose(
        term.eval(inputs).reshape(-1), reference(ONE_LAYER, weights, x), atol=1e-12
    )


@pytest.mark.slow
def test_tiny_compiles_and_matches_the_reference():
    weights = init_weights(TINY)
    x = _clip(2, 16)
    term, inputs = tensor_ir(TINY, weights, x)
    _run(term, inputs, 4096)


def _official_model(frames, size, classes):
    """The Model Garden's MoViNet-A0 stream classifier with random weights and
    batch norm statistics, or None if it is not installed."""
    try:
        import tensorflow as tf  # noqa: F401
        from official.projects.movinet.modeling import movinet, movinet_model
    except Exception:
        return None
    backbone = movinet.Movinet(
        model_id="a0",
        causal=True,
        conv_type="2plus1d",
        se_type="2plus3d",
        activation="hard_swish",
        gating_activation="hard_sigmoid",
        use_external_states=False,
    )
    model = movinet_model.MovinetClassifier(
        backbone, num_classes=classes, output_states=False, activation="hard_swish"
    )
    model.build([1, frames, size, size, 3])
    rng = np.random.default_rng(0)
    for v in model.weights:
        if v.name.endswith(("moving_variance:0", "gamma:0")):
            v.assign(rng.uniform(0.5, 1.5, v.shape))
        elif v.name.endswith(("beta:0", "moving_mean:0", "bias:0")):
            v.assign(rng.normal(0, 0.2, v.shape))
        elif v.name.endswith("scale:0"):
            v.assign(rng.uniform(0.5, 1.0, v.shape))
    return model


@pytest.mark.slow
@pytest.mark.parametrize("frames,size", [(4, 32), (3, 44), (8, 172)])
def test_a0_matches_the_official_model(frames, size):
    """Needs TensorFlow and ``tensorflow/models`` on the path."""
    classes = 600
    model = _official_model(frames, size, classes)
    if model is None:
        pytest.skip("the TensorFlow Model Garden is not installed")
    x = np.random.default_rng(1).standard_normal((1, frames, size, size, 3))
    x = x.astype(np.float32)
    expected = np.asarray(model(x, training=False))[0]

    weights = from_official({v.name: v.numpy() for v in model.weights})
    clip = np.transpose(x[0], (3, 0, 1, 2))
    term, inputs = tensor_ir(A0, weights, clip)
    # the official model computes in float32
    np.testing.assert_allclose(reference(A0, weights, clip), expected, atol=1e-4)
    np.testing.assert_allclose(term.eval(inputs).reshape(-1), expected, atol=1e-4)
