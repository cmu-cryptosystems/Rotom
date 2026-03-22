"""ResNet-20 SiLU polynomial path on the Toy backend.

Heavy end-to-end test (opt-in via ``ROTOM_RUN_HEAVY_E2E=1``) builds the **full**
``build_resnet20_silu_poly_graph`` (stem through FC), with ``n=32768``. Runtime is
dominated by the number of HE ops × vector width, not Python recursion: Toy’s hot
path walks each ciphertext DAG in iterative post-order; ``Toy.eval`` may recurse
only on occasional ``HEOp.CS`` edges when a child is missing from ``env``.

**Memory:** use-count eviction lowers peak versus keeping all intermediates, but
RSS can still grow large (many length-``n`` vectors live at once inside one DAG).
If ``Maximum resident set size`` from ``/usr/bin/time -v`` sits at your cgroup
``MemoryMax`` and you see ``Command terminated by signal 9``, the limit—not a
Python leak—is the usual cause.

- **Stem** (conv + BN + SiLU poly): Toy matches ``tensor_ir.eval`` under the usual
  ``check_results`` tolerances.
- **Full-graph heavy test** below: layout → lower → Toy over the **entire** SiLU-poly
  ResNet-20 graph; same ``check_results`` / ``allclose`` tolerances as the stem test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet import resnet20_tensor_ir as R
from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    build_resnet20_silu_poly_graph,
    populate_resnet20_no_activation_inputs,
)
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.test_util import get_default_args
from util.checker import check_results

_RESNET_TOY_N = 32768
_SKIP_ON_CI = os.environ.get("CI", "").strip().lower() in {"1", "true", "yes", "on"}
_RUN_HEAVY_E2E = os.environ.get("ROTOM_RUN_HEAVY_E2E", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@pytest.mark.slow
def test_resnet20_silu_poly_stem_toy_matches_tensor_eval() -> None:
    """Stem + BN + SiLU poly through Toy matches tensor eval (known-good prefix)."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_no_activation_inputs(model, inputs)

    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    h, w = 32, 32
    t = TensorTerm.Tensor("input", [3, 32, 32], True)
    t = R._conv_same(t, "conv1_w", inputs, 1)
    h, w = R._spatial_hw_after_conv3(h, w, 1)
    t = R._bn_affine_hw(t, "bn1", 16, h, w, inputs)
    t = R._silu_poly(t)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_poly_stem"

    kernel = LayoutAssignment(t, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(t, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
@pytest.mark.skipif(
    _SKIP_ON_CI or not _RUN_HEAVY_E2E,
    reason="too heavy (memory/time) for default test runs; set ROTOM_RUN_HEAVY_E2E=1 to opt in",
)
def test_resnet20_silu_poly_toy_matches_tensor_eval() -> None:
    """Full SiLU-poly ResNet-20 tensor graph: layout, lower, Toy, then ``check_results``.

    ``args.benchmark`` names the l2_0 scenario for assignment heuristics, but
    ``tensor_ir`` is the full graph from :func:`build_resnet20_silu_poly_graph`.
    """
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    # `apply_layout` spends significant time rebuilding a layout->indices plan.
    # Persist those plans on disk so repeated runs (and CI) don't pay the
    # same startup cost.
    os.environ.setdefault(
        "ROTOM_APPLY_LAYOUT_PLAN_CACHE_DIR",
        str(
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "rotom"
            / "apply_layout_plans"
        ),
    )

    inputs: dict = {}
    populate_resnet20_no_activation_inputs(model, inputs)

    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    tensor_ir = build_resnet20_silu_poly_graph(inputs)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_poly_l2_0"

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()

    check_results(tensor_ir, inputs, kernel, backend_results, 0, args)
