"""
Build deterministic TENSOR seed kernels and evaluate transformed kernels (Lower + Toy + cost).

This path is intentionally restricted to **leaf TENSOR** kernels (no CS children) so
layout edits do not desynchronize a kernel DAG. Matmul / binop alignment is enforced
by Rotom's full assignment pass; exploring that space belongs in ``layout_strategy.py``.

For higher-level op alignment, combine this harness with kernels produced only after
``LayoutAssignment`` on subgraphs that remain valid under your transform — not wired here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np

from assignment.gen.gen_tensor import gen_tensor
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from lower.lower import Lower
from util.layout_util import apply_layout


def build_seed_tensor_kernel(
    *,
    name: str = "x",
    shape: tuple[int, ...] = (8, 8),
    n_slots: int = 4096,
    secret: bool = True,
) -> tuple[TensorTerm, Kernel, tuple[int, ...]]:
    """Canonical smallest-layout seed among ``gen_tensor`` candidates (2D fits in n)."""
    term = TensorTerm.Tensor(name, list(shape), secret)
    kernels = gen_tensor(term, secret, list(shape), n_slots)
    seed = min(kernels, key=lambda k: k.layout.layout_str())
    assert seed.op == KernelOp.TENSOR
    sh = tuple(int(x) for x in shape)
    return term, seed, sh


def load_propose_fn(program_path: str) -> Callable[[Kernel], Kernel]:
    spec = importlib.util.spec_from_file_location("layout_transform_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "propose_transformed_kernel"):
        raise AttributeError("program must define propose_transformed_kernel(seed_kernel: Kernel) -> Kernel")
    return mod.propose_transformed_kernel


def _same_tensor_term(a: TensorTerm, b: TensorTerm) -> bool:
    return a.op == b.op and a.cs == b.cs


def evaluate_transformed_kernel(
    term: TensorTerm,
    seed: Kernel,
    shape: tuple[int, ...],
    propose: Callable[[Kernel], Kernel],
    *,
    network: str = "lan",
) -> dict[str, Any]:
    """
    Lower ``propose(seed)``, run Toy, compare to plaintext layout eval; return cost + flags.

    Returns keys: ok, semantic_ok, total_cost, error (optional), layout_str_{before,after}.
    """
    from argparse import Namespace

    args = Namespace(
        backend="toy",
        n=seed.layout.n,
        benchmark="layout_transform",
        microbenchmark="main",
        rolls=False,
        strassens=False,
        net=network,
        cache=False,
        serialize=False,
        mock=False,
        fuzz=False,
        fuzz_result=False,
        conv_roll=False,
        fn="layout_transform",
        not_secure=False,
        skip_toy_eval_checks=False,
    )

    out: dict[str, Any] = {
        "layout_str_before": seed.layout.layout_str(),
        "ok": False,
        "semantic_ok": False,
        "total_cost": float("inf"),
    }

    try:
        k2 = propose(seed)
    except Exception as e:
        out["error"] = f"propose_transformed_kernel: {type(e).__name__}: {e}"
        return out

    if not isinstance(k2, Kernel):
        out["error"] = "propose_transformed_kernel must return a Kernel"
        return out

    if k2.op != KernelOp.TENSOR or not _same_tensor_term(k2.layout.term, term):
        out["error"] = "transformed kernel must stay TENSOR with same tensor term"
        return out

    out["layout_str_after"] = k2.layout.layout_str()

    try:
        circuit = Lower(k2).run()
    except Exception as e:
        out["error"] = f"Lower: {type(e).__name__}: {e}"
        return out

    np.random.seed(0)
    inputs = {term.cs[0]: np.random.randn(*shape).astype(np.float64)}

    try:
        results = Toy(circuit, inputs, args).run()
    except Exception as e:
        out["error"] = f"Toy: {type(e).__name__}: {e}"
        return out

    expected_cts = apply_layout(term.eval(inputs), k2.layout)
    semantic_ok = True
    max_diff = 0.0
    for expected, result in zip(expected_cts, results):
        if not np.allclose(expected, result, rtol=1e-2, atol=1e-2):
            semantic_ok = False
            diff = np.asarray(expected) - np.asarray(result)
            max_diff = max(max_diff, float(np.max(np.abs(diff))))

    out["semantic_ok"] = semantic_ok
    out["max_diff"] = max_diff
    out["total_cost"] = float(KernelCost(k2, network).total_cost())
    out["ok"] = semantic_ok
    return out


def baseline_program_path() -> Path:
    return Path(__file__).resolve().parent / "layout_transform_program.py"
