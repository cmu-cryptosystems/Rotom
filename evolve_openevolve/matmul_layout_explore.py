"""
Explore matrix-multiplication kernels over **layout representation + alignment rules**.

This module does **not** evolve code; it drives the same path as
``assignment/gen/gen_binop.py`` → ``replicate_dimensions`` → ``match_layout`` →
``output_layout`` so every row is a **valid** (aligned) MATMUL kernel Rotom would
consider.

Alignment (2D @ 2D)
-------------------
``assignment.alignment.get_dim_alignment`` pairs logical tensor axes as::

    {(0, None), (1, 0), (None, 1)}

meaning: A's dim 0 is free (batch row), A's dim 1 aligns with B's dim 0
(contraction / inner), B's dim 1 is free (output column). Extents along
aligned axes must match after padding (``Shape.padded_shapes``).

Further checks live in ``gen_align.check_alignment`` (dimension extents along
the alignment graph + ``check_roll_alignment`` for MATMUL rolls).

Usage::

    python -m evolve_openevolve.matmul_layout_explore --m 8 --k 8 --n 8 --top 20
    python -m evolve_openevolve.matmul_layout_explore --json --m 4 --k 4 --n 4

Environment: run from **Rotom** repo root with ``PYTHONPATH=``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from assignment.gen.gen_binop import gen_binop
from assignment.gen.gen_tensor import gen_tensor
from frontends.tensor import TensorOp, TensorTerm
from ir.analysis.secret import Secret
from ir.analysis.shape import Shape
from ir.kernel import KernelOp
from ir.kernel_cost import KernelCost
from ir.layout_utils import dimension_merging


def matmul_alignment_pairs_2d2d() -> set[tuple[int | None, int | None]]:
    """Logical alignment for ``(M,K) @ (K,N)`` matmul (same as ``get_dim_alignment``)."""
    return {(0, None), (1, 0), (None, 1)}


def sorted_alignment_pairs_2d2d() -> list[tuple[int | None, int | None]]:
    """Deterministic order for display (None sorts before integers in second key)."""
    align = matmul_alignment_pairs_2d2d()
    return sorted(
        align,
        key=lambda t: (
            t[0] is not None,
            t[0] if t[0] is not None else -1,
            t[1] is not None,
            t[1] if t[1] is not None else -1,
        ),
    )


@dataclass(frozen=True)
class MatmulLayoutOption:
    """One feasible MATMUL kernel after alignment."""

    output_layout_str: str
    operand_a_layout_str: str
    operand_b_layout_str: str
    total_cost_lan: float
    rolls_flag_used: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def explore_matmul_kernels(
    *,
    m: int,
    k: int,
    n_out: int,
    n_slots: int,
    roll_flag: bool,
    network: str = "lan",
) -> list[MatmulLayoutOption]:
    """
    Enumerate MATMUL kernels for ``Tensor(m,k) @ Tensor(k,n_out)`` using padded shapes.

    Candidate input layouts come from ``gen_tensor`` on each operand; combinations flow
    through ``gen_binop`` (replication, optional sum-rolls, ``match_layout``,
    ``output_layout``). Each returned option is alignment-valid per Rotom.
    """
    if m <= 0 or k <= 0 or n_out <= 0:
        raise ValueError("m, k, n_out must be positive")

    a = TensorTerm.Tensor("a", [m, k], True)
    b = TensorTerm.Tensor("b", [k, n_out], True)
    term = a @ b
    if term.op != TensorOp.MATMUL:
        raise RuntimeError("expected MATMUL term")

    comp = term
    Secret(comp).run()
    shape = Shape(comp)
    shape.run()

    pa = shape.padded_shapes[a]
    pb = shape.padded_shapes[b]

    kernels_a = gen_tensor(a, True, pa, n_slots)
    kernels_b = gen_tensor(b, True, pb, n_slots)

    cs_kernels = [kernels_a, kernels_b]
    shapes = [pa, pb]

    out_map = gen_binop(term, cs_kernels, shapes, roll_flag)
    raw = out_map.get(term, set())
    options: list[MatmulLayoutOption] = []

    for kernel in raw:
        if kernel.op != KernelOp.MATMUL:
            continue
        merged = dimension_merging(kernel.layout)
        out_str = merged.layout_str()
        a_str = dimension_merging(kernel.cs[0].layout).layout_str()
        b_str = dimension_merging(kernel.cs[1].layout).layout_str()
        cost = float(KernelCost(kernel, network).total_cost())
        options.append(
            MatmulLayoutOption(
                output_layout_str=out_str,
                operand_a_layout_str=a_str,
                operand_b_layout_str=b_str,
                total_cost_lan=cost,
                rolls_flag_used=roll_flag,
            )
        )

    options.sort(key=lambda o: (o.total_cost_lan, o.output_layout_str))
    return options


def dedupe_matmul_options(opts: Iterable[MatmulLayoutOption]) -> list[MatmulLayoutOption]:
    seen: set[tuple[str, str, str]] = set()
    out: list[MatmulLayoutOption] = []
    for o in opts:
        key = (o.operand_a_layout_str, o.operand_b_layout_str, o.output_layout_str)
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--m", type=int, default=8, help="A rows (inner dim K)")
    p.add_argument("--k", type=int, default=8, help="A cols / B rows")
    p.add_argument("--n", type=int, default=8, dest="n_out", help="B cols")
    p.add_argument("--slots", type=int, default=4096, help="HE vector size n")
    p.add_argument(
        "--no-rolls",
        action="store_true",
        help="set roll_flag=False (smaller search, no roll_dimensions path)",
    )
    p.add_argument("--top", type=int, default=30, help="print only cheapest N")
    p.add_argument("--json", action="store_true", help="print JSON array to stdout")
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="collapse duplicate (A,B,out) layout triples",
    )
    args = p.parse_args(argv)

    opts = explore_matmul_kernels(
        m=args.m,
        k=args.k,
        n_out=args.n_out,
        n_slots=args.slots,
        roll_flag=not args.no_rolls,
    )
    if args.dedupe:
        opts = dedupe_matmul_options(opts)

    if args.json:
        print(json.dumps([o.to_json_dict() for o in opts[: args.top]], indent=2))
        return

    print("MATMUL alignment (2D @ 2D):", sorted_alignment_pairs_2d2d())
    print(
        f"Shapes (logical) A=[{args.m},{args.k}] B=[{args.k},{args.n_out}] "
        f"n_slots={args.slots} roll_flag={not args.no_rolls}"
    )
    print(f"Enumerated {len(opts)} kernels (showing up to {args.top} cheapest by LAN cost)\n")
    for i, o in enumerate(opts[: args.top], 1):
        print(f"--- {i} cost={o.total_cost_lan:.4f} ---")
        print(f"  A: {o.operand_a_layout_str}")
        print(f"  B: {o.operand_b_layout_str}")
        print(f"  out: {o.output_layout_str}")


if __name__ == "__main__":
    main()
