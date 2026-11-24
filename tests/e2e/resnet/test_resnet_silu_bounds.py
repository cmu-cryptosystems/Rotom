"""Per-site SiLU ranges for the DaCapo ResNet-20 checkpoint."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import benchmarks.e2e.resnet.resnet20_tensor_ir as R
from benchmarks.e2e.resnet.resnet_data import CKPT_FILE, cifar10_test_dataset
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet.silu_poly_calibration import (
    iter_resnet20_silu_modules,
    resnet20_silu_site_names,
)
from frontends.tensor import TensorOp
from frontends.tensor_args import PolyCallArgs
from util.silu_polycall_eval import (
    DEFAULT_SILU_POLY_DEGREE,
    SILU_POLY_DEGREE_KEY,
    silu_poly_ascending_coeffs,
)


def _site_bounds(tensor_ir):
    """The (lo, hi) of each SiLU in the program, in forward order."""
    bounds = []
    for term in tensor_ir.post_order():
        if term.op == TensorOp.POLY_CALL:
            args = PolyCallArgs.from_term(term)
            bounds.append((args.lower_bound, args.upper_bound))
    return bounds


class _PolySiLU(nn.Module):
    """``x * q(x)`` with the evaluator's coefficients, in the input's precision."""

    def __init__(self, lo, hi, degree, clip):
        super().__init__()
        self.lo, self.hi, self.clip = lo, hi, clip
        self.coeffs = silu_poly_ascending_coeffs(lo, hi, degree, 80)

    def forward(self, x):
        if self.clip:
            x = x.clamp(self.lo, self.hi)
        q = torch.zeros_like(x)
        for c in reversed(self.coeffs):
            q = q * x + c
        return x * q


def _with_polynomials(model, bounds, degree, clip):
    acts = [_PolySiLU(lo, hi, degree, clip) for lo, hi in bounds]
    model.act_conv1 = acts[0]
    k = 1
    for layer in (model.layer1, model.layer2, model.layer3):
        for block in layer:
            block.act1, block.act2 = acts[k], acts[k + 1]
            k += 2
    return model


def _test_images(count):
    try:
        dataset = cifar10_test_dataset()
    except (FileNotFoundError, ModuleNotFoundError) as e:
        pytest.skip(str(e))
    images = torch.stack([dataset[i][0] for i in range(count)]).double()
    labels = torch.tensor([dataset[i][1] for i in range(count)])
    return images, labels


def _accuracy(model, images, labels):
    with torch.no_grad():
        predicted = torch.cat(
            [model(images[i : i + 250]) for i in range(0, len(images), 250)]
        ).argmax(1)
    return float((predicted == labels).double().mean())


def test_bounds_name_every_silu_site() -> None:
    assert list(R.RESNET20_DACAPO_SILU_BOUNDS) == resnet20_silu_site_names()
    assert all(lo < 0 < hi for lo, hi in R.RESNET20_DACAPO_SILU_BOUNDS.values())


def test_each_site_is_fitted_on_its_own_range() -> None:
    model = resnet20(num_classes=10)
    assert len(list(iter_resnet20_silu_modules(model))) == 19

    inputs: dict = {}
    R.populate_resnet20_inputs(
        model,
        inputs,
        silu_bounds=R.RESNET20_DACAPO_SILU_BOUNDS,
        silu_degree=R.RESNET20_DACAPO_SILU_DEGREE,
    )
    tensor_ir = R.build_resnet20_silu_poly_graph(inputs)

    assert _site_bounds(tensor_ir) == list(R.RESNET20_DACAPO_SILU_BOUNDS.values())
    assert inputs[SILU_POLY_DEGREE_KEY] == R.RESNET20_DACAPO_SILU_DEGREE


def test_sites_share_the_global_range_without_bounds() -> None:
    inputs: dict = {}
    R.populate_resnet20_inputs(resnet20(num_classes=10), inputs)
    tensor_ir = R.build_resnet20_silu_poly_graph(inputs)

    assert _site_bounds(tensor_ir) == [R.GLOBAL_SILU_BOUNDS] * 19
    assert inputs[SILU_POLY_DEGREE_KEY] == DEFAULT_SILU_POLY_DEGREE


def test_a_prefix_of_the_network_uses_the_same_sites() -> None:
    inputs: dict = {}
    R.populate_resnet20_inputs(
        resnet20(num_classes=10), inputs, silu_bounds=R.RESNET20_DACAPO_SILU_BOUNDS
    )
    expected = list(R.RESNET20_DACAPO_SILU_BOUNDS.values())

    through_l2 = R.build_resnet20_silu_poly_graph_to_depth(inputs, "l2")
    assert _site_bounds(through_l2) == expected[:13]

    inputs["l2_0_block_in"] = np.zeros((16, 32, 32))
    l2_0 = R.build_resnet20_silu_poly_l2_0_block_graph(inputs)
    assert _site_bounds(l2_0) == expected[7:9]


@pytest.mark.e2e
@pytest.mark.slow
def test_dacapo_checkpoint_keeps_its_accuracy_with_polynomial_silu() -> None:
    """No clipping here: an encrypted evaluation cannot clip its input."""
    if not os.path.exists(CKPT_FILE):
        pytest.skip(f"Missing checkpoint at {CKPT_FILE}")
    images, labels = _test_images(1000)

    exact = _accuracy(R.populate_dacapo_resnet20_inputs({}).double(), images, labels)
    per_site = _accuracy(
        _with_polynomials(
            R.populate_dacapo_resnet20_inputs({}).double(),
            R.RESNET20_DACAPO_SILU_BOUNDS.values(),
            R.RESNET20_DACAPO_SILU_DEGREE,
            clip=False,
        ),
        images,
        labels,
    )
    one_fit = _accuracy(
        _with_polynomials(
            R.populate_dacapo_resnet20_inputs({}).double(),
            [R.GLOBAL_SILU_BOUNDS] * 19,
            DEFAULT_SILU_POLY_DEGREE,
            clip=False,
        ),
        images,
        labels,
    )

    assert exact > 0.90
    assert per_site > exact - 0.01
    assert one_fit < 0.75


@pytest.mark.e2e
@pytest.mark.slow
def test_tensor_eval_matches_pytorch_with_the_same_polynomials() -> None:
    """The accuracy measured in PyTorch is the accuracy of the tensor program."""
    if not os.path.exists(CKPT_FILE):
        pytest.skip(f"Missing checkpoint at {CKPT_FILE}")
    images, _ = _test_images(2)

    inputs: dict = {}
    model = R.populate_dacapo_resnet20_inputs(inputs).double()
    tensor_ir = R.build_resnet20_silu_poly_graph(inputs)
    # the evaluator clips each SiLU input to its range
    model = _with_polynomials(
        model,
        R.RESNET20_DACAPO_SILU_BOUNDS.values(),
        R.RESNET20_DACAPO_SILU_DEGREE,
        clip=True,
    )

    for image in images:
        inputs["input"] = image.numpy()
        got = np.asarray(tensor_ir.eval(inputs), dtype=np.float64).ravel()[:10]
        with torch.no_grad():
            want = model(image[None]).numpy().ravel()
        # the two sum the polynomial's sixteen terms in a different order
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
