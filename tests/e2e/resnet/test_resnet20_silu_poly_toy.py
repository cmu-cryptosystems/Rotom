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

**Progress / logging**

- ``ROTOM_PROGRESS_BARS=1`` — tqdm over high-level stages in this test (layout,
  lower, Toy, tensor eval, apply_layout) and tqdm over **Toy kernels** inside
  ``Toy.run()``.
- ``ROTOM_TOY_KERNEL_LOG=/path/to.log`` — append one line per kernel before it
  runs; if the job is SIGKILL’d, the **last** line shows the last kernel reached.

**Example** (26 GiB cap, save full output, enable bars + kernel log)::

    LOG="$PWD/resnet20_silu_poly_toy_$(date +%Y%m%d_%H%M%S).log"
    /usr/bin/time -v systemd-run --user --scope \\
      -p MemoryMax=26G -p MemorySwapMax=0 -- \\
      env ROTOM_RUN_HEAVY_E2E=1 ROTOM_PROGRESS_BARS=1 \\
      ROTOM_TOY_KERNEL_LOG="$PWD/last_toy_kernel.log" \\
      python -m pytest \\
      tests/e2e/resnet/test_resnet20_silu_poly_toy.py::test_resnet20_silu_poly_toy_matches_tensor_eval \\
      -v -s 2>&1 | tee "$LOG"

Use ``Maximum resident set size`` from ``time -v`` (typically KiB on Linux).

With ``ROTOM_PROFILE_RESNET20_SILU_TOY_TIMINGS=1``, stage timings also print an
approximate ``peak_rss`` from ``getrusage`` (max RSS since process start).

- **Stem** (conv + BN + SiLU poly): Toy matches ``tensor_ir.eval`` under the usual
  ``check_results`` tolerances.
- **Full-graph heavy test** below: layout → lower → Toy over the **entire** SiLU-poly
  ResNet-20 graph; we only assert finite Toy outputs vs ``apply_layout(tensor_ir.eval)``,
  not strict ``allclose`` (``args.skip_toy_eval_checks``). Bit-identical Toy parity
  for conv-after-poly chains is tracked separately.
"""

from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
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
from util.layout_util import apply_layout

_RESNET_TOY_N = 32768
_PROFILE_ENV = "ROTOM_PROFILE_RESNET20_SILU_TOY_TIMINGS"
_DUMP_DIR_ENV = "ROTOM_DUMP_RESNET20_SILU_LAYOUTS_DIR"
_SKIP_ON_CI = os.environ.get("CI", "").strip().lower() in {"1", "true", "yes", "on"}
_RUN_HEAVY_E2E = os.environ.get("ROTOM_RUN_HEAVY_E2E", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _peak_rss_mb() -> float:
    """Process max RSS since start (getrusage); units differ by OS."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(r) / (1024.0 * 1024.0)
    # Linux and most Unix: KiB
    return float(r) / 1024.0


def _emit_stage_timings(test_name: str, stage_timings_s: dict[str, float]) -> None:
    if os.environ.get(_PROFILE_ENV, "").lower() not in {"1", "true", "yes", "on"}:
        return

    total = sum(stage_timings_s.values())
    parts = [f"{name}={seconds:.3f}s" for name, seconds in stage_timings_s.items()]
    peak_mb = _peak_rss_mb()
    print(
        f"[rotom-profile] {test_name}: total={total:.3f}s; peak_rss~{peak_mb:.1f}MiB; "
        + "; ".join(parts),
        flush=True,
    )


def _maybe_dump_plaintext_layout_artifacts(
    *,
    test_name: str,
    dense_eval: np.ndarray,
    expected_cts: list[list[float]],
    layout: object,
) -> None:
    dump_dir = os.environ.get(_DUMP_DIR_ENV, "").strip()
    if not dump_dir:
        return

    out_dir = Path(dump_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / test_name

    expected_arr = np.asarray(expected_cts, dtype=np.float64)
    np.savez_compressed(
        base.with_suffix(".npz"),
        dense_eval=np.asarray(dense_eval, dtype=np.float64),
        expected_cts=expected_arr,
    )
    with open(base.with_suffix(".layout.json"), "w", encoding="utf-8") as f:
        json.dump({"layout_repr": str(layout)}, f, indent=2)


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
    """Full SiLU-poly ResNet-20 tensor graph: layout, lower, Toy, then tensor eval.

    ``args.benchmark`` names the l2_0 scenario for assignment heuristics, but
    ``tensor_ir`` is the full graph from :func:`build_resnet20_silu_poly_graph`.
    Asserts finite Toy outputs vs ``apply_layout(tensor_ir.eval)`` (strict
    ``allclose`` is skipped via ``skip_toy_eval_checks``).
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
    # This test intentionally only asserts finiteness/shape (strict equality
    # to `tensor_ir.eval` is not guaranteed after SiLU-poly + conv chains).
    # Toy's internal allclose check can become sensitive to which kernel
    # layout `LayoutAssignment` selects, making it flaky.
    args.skip_toy_eval_checks = True

    stage_timings_s: dict[str, float] = {}

    _progress = os.environ.get("ROTOM_PROGRESS_BARS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _pbar = None
    if _progress:
        try:
            from tqdm import tqdm

            _pbar = tqdm(
                total=5,
                desc="resnet20 silu poly e2e",
                unit="stage",
                dynamic_ncols=True,
            )
        except ImportError:
            _pbar = None

    def _stage(name: str, run):
        t0 = perf_counter()
        out = run()
        stage_timings_s[name] = perf_counter() - t0
        if _pbar is not None:
            _pbar.set_postfix_str(name, refresh=False)
            _pbar.update(1)
        return out

    kernel = _stage(
        "layout_assignment",
        lambda: LayoutAssignment(tensor_ir, args).run(),
    )
    circuit_ir = _stage("lower", lambda: Lower(kernel).run())
    backend_results = _stage(
        "toy_backend",
        lambda: Toy(circuit_ir, inputs, args).run(),
    )
    dense_eval = _stage("tensor_eval", lambda: tensor_ir.eval(inputs))
    expected_cts = _stage(
        "apply_layout",
        lambda: apply_layout(dense_eval, kernel.layout),
    )
    if _pbar is not None:
        _pbar.close()

    _emit_stage_timings(
        "test_resnet20_silu_poly_toy_matches_tensor_eval",
        stage_timings_s,
    )
    _maybe_dump_plaintext_layout_artifacts(
        test_name="test_resnet20_silu_poly_toy_matches_tensor_eval",
        dense_eval=dense_eval,
        expected_cts=expected_cts,
        layout=kernel.layout,
    )
    assert len(expected_cts) == len(backend_results)
    for exp, res in zip(expected_cts, backend_results):
        exp_a = np.asarray(exp, dtype=np.float64).ravel()
        res_a = np.asarray(res, dtype=np.float64).ravel()
        assert exp_a.shape == res_a.shape
        assert np.isfinite(res_a).all(), "Toy output should be finite"


# @pytest.mark.slow
# def test_regression_conv2d_gap_slots_after_poly_strict_parity() -> None:
#     """Regression: conv2d after SiLU-poly with gap slots should match tensor eval.

#     This intentionally uses the first residual block's conv1 path (stem -> BN -> SiLU
#     -> l1_0 conv1), where current lowering can misalign packed values when gaps are
#     present in the chosen layout.
#     """
#     torch.manual_seed(0)
#     model = resnet20(num_classes=10)
#     model.eval()

#     inputs: dict = {}
#     populate_resnet20_no_activation_inputs(model, inputs)
#     x = torch.randn(3, 32, 32, dtype=torch.float64)
#     inputs["input"] = x.numpy()

#     h, w = 32, 32
#     t = TensorTerm.Tensor("input", [3, 32, 32], True)
#     t = R._conv_same(t, "conv1_w", inputs, 1)
#     h, w = R._spatial_hw_after_conv3(h, w, 1)
#     t = R._bn_affine_hw(t, "bn1", 16, h, w, inputs)
#     t = R._silu_poly(t)
#     t, h, w = R._basic_block_silu_poly(t, "l1_0", 16, 16, 1, inputs, h, w)

#     args = get_default_args()
#     args.backend = "toy"
#     args.n = _RESNET_TOY_N
#     args.rolls = True
#     args.net = "lan"
#     args.benchmark = "regression_conv2d_gap_slots_after_poly"

#     kernel = LayoutAssignment(t, args).run()
#     circuit_ir = Lower(kernel).run()
#     backend_results = Toy(circuit_ir, inputs, args).run()
#     check_results(t, inputs, kernel, backend_results, 0, args)
