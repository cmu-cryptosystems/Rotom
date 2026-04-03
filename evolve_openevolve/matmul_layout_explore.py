"""
Explore matrix-multiplication kernels over **layout representation + alignment rules**.

**Enumeration** (default) replays ``assignment/gen/gen_binop.py`` end-to-end: every row
is a MATMUL kernel Rotom would already consider for the given seeds.

**Local search** (``--local-search``) goes further: starting from those kernels, it
applies **operand-only** moves (permute ``layout.dims``, append ``Roll`` via
``layout_transform_lib``), then recomputes the MATMUL output with
``gen_align.output_layout`` and ``check_alignment``. Use ``--neighbor-mode wide`` to
try all traversal permutations when the dim count is small (≤6). Per-base budgets:
``--max-attempts-per-seed``. Optional ``--verify-lower`` runs ``Lower()`` on the
printed top kernels (structural sanity; not full Toy matmul semantics).

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
    python -m evolve_openevolve.matmul_layout_explore --local-search --neighbor-mode wide --m 4 --k 4 --n 4 --slots 256
    python -m evolve_openevolve.matmul_layout_explore --local-search --verify-lower --top 10 --m 4 --k 4 --n 4 --slots 256 --no-rolls
    python -m evolve_openevolve.matmul_layout_explore --json --m 4 --k 4 --n 4

Environment: run from **Rotom** repo root with ``PYTHONPATH=``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

from assignment.alignment import get_dim_alignment
from assignment.gen.gen_align import check_alignment, output_layout
from assignment.gen.gen_binop import gen_binop
from assignment.gen.gen_tensor import gen_tensor
from frontends.tensor import TensorOp, TensorTerm
from ir.analysis.secret import Secret
from ir.analysis.shape import Shape
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout_utils import dimension_merging
from lower.lower import Lower

from evolve_openevolve.layout_transform_lib import (
    apply_append_roll,
    apply_permute_traversal_dims,
)


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
    source: str = "enumerate"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matmul_term_and_shapes(
    m: int, k: int, n_out: int
) -> tuple[TensorTerm, TensorTerm, TensorTerm, list[Any], list[Any], set]:
    """Build ``a @ b``, run analyses, return (term, a, b, pa, pb, alignment)."""
    a = TensorTerm.Tensor("a", [m, k], True)
    b = TensorTerm.Tensor("b", [k, n_out], True)
    term = a @ b
    if term.op != TensorOp.MATMUL:
        raise RuntimeError("expected MATMUL term")
    Secret(term).run()
    shape = Shape(term)
    shape.run()
    pa = shape.padded_shapes[a]
    pb = shape.padded_shapes[b]
    alignment = get_dim_alignment(term, [pa, pb])
    return term, a, b, pa, pb, alignment


def _operand_neighborhood_kernels(
    kernel: Kernel,
    *,
    mode: Literal["default", "wide"] = "default",
    wide_perm_max_dims: int = 6,
) -> Iterable[Kernel]:
    """
    Yield Kernels with the same op/cs tree but perturbed root layout (permute / roll).

    ``default``: cyclic shifts of ``layout.dims`` plus every ``apply_append_roll(i,j)``.
    ``wide``: all ``d!`` traversal permutations when ``d <= wide_perm_max_dims``; otherwise
    same permutations as ``default``. Roll moves are always included.
    """
    layout = kernel.layout
    d = len(layout.dims)

    if mode == "wide" and d <= wide_perm_max_dims:
        for perm in itertools.permutations(range(d)):
            try:
                nl = apply_permute_traversal_dims(layout, perm)
                yield Kernel(kernel.op, kernel.cs, nl)
            except (ValueError, AssertionError):
                pass
    else:
        for shift in range(d):
            perm = tuple((i + shift) % d for i in range(d))
            try:
                nl = apply_permute_traversal_dims(layout, perm)
                yield Kernel(kernel.op, kernel.cs, nl)
            except (ValueError, AssertionError):
                pass

    for i, j in itertools.permutations(range(d), 2):
        try:
            nl = apply_append_roll(layout, i, j)
            yield Kernel(kernel.op, kernel.cs, nl)
        except (ValueError, AssertionError, TypeError):
            pass


def try_lower_matmul_kernel(kernel: Kernel) -> dict[str, Any]:
    """
    Structural check: ``Lower(kernel)`` succeeds (circuit IR builds).

    Full Toy semantic checks for MATMUL are environment-sensitive; use this as a fast
    filter that the lowered graph is internally consistent.
    """
    try:
        Lower(kernel).run()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def collect_matmul_kernel_objects(
    *,
    m: int,
    k: int,
    n_out: int,
    n_slots: int,
    roll_flag: bool,
) -> tuple[TensorTerm, set, list[Kernel]]:
    """Single ``gen_binop`` pass; returns MATMUL kernels plus term and alignment."""
    if m <= 0 or k <= 0 or n_out <= 0:
        raise ValueError("m, k, n_out must be positive")
    term, a, b, pa, pb, alignment = _matmul_term_and_shapes(m, k, n_out)
    kernels_a = gen_tensor(a, True, pa, n_slots)
    kernels_b = gen_tensor(b, True, pb, n_slots)
    out_map = gen_binop(term, [kernels_a, kernels_b], [pa, pb], roll_flag)
    raw = out_map.get(term, set())
    matmuls = [x for x in raw if x.op == KernelOp.MATMUL]
    return term, alignment, matmuls


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
    _term, _align, kernels = collect_matmul_kernel_objects(
        m=m, k=k, n_out=n_out, n_slots=n_slots, roll_flag=roll_flag
    )
    options: list[MatmulLayoutOption] = []

    for kernel in kernels:
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
                source="enumerate",
            )
        )

    options.sort(key=lambda o: (o.total_cost_lan, o.output_layout_str))
    return options


def local_search_matmul_kernels(
    *,
    m: int,
    k: int,
    n_out: int,
    n_slots: int,
    roll_flag: bool,
    network: str = "lan",
    max_local_attempts: int | None = 800,
    neighbor_mode: Literal["default", "wide"] = "default",
    max_attempts_per_seed: int | None = None,
    collect_kernel_refs: bool = False,
) -> list[MatmulLayoutOption] | tuple[list[MatmulLayoutOption], list[Kernel]]:
    """
    Expand beyond raw ``gen_binop`` enumeration: for each base MATMUL kernel, try
    operand layout neighbors (see ``_operand_neighborhood_kernels``). Valid neighbors
    must pass ``check_alignment``; outputs are built with ``output_layout`` and
    scored with ``KernelCost``.

    ``max_local_attempts`` caps total neighbor *trials* (each trial counts before
    alignment). ``None`` or ``<= 0`` means no global cap.

    ``max_attempts_per_seed`` caps trials **per base kernel** (same counting).

    ``collect_kernel_refs``: if True, return ``(options, kernels)`` where ``kernels[i]``
    is the root ``Kernel`` for ``options[i]`` (for ``try_lower_matmul_kernel``, etc.).
    """
    term, alignment, base_kernels = collect_matmul_kernel_objects(
        m=m, k=k, n_out=n_out, n_slots=n_slots, roll_flag=roll_flag
    )

    seen: set[tuple[str, str, str]] = set()
    options: list[MatmulLayoutOption] = []
    kernel_refs: list[Kernel] = []

    def add_from_kernel(k: Kernel, source: str) -> None:
        merged = dimension_merging(k.layout)
        out_str = merged.layout_str()
        a_str = dimension_merging(k.cs[0].layout).layout_str()
        b_str = dimension_merging(k.cs[1].layout).layout_str()
        key = (a_str, b_str, out_str)
        if key in seen:
            return
        seen.add(key)
        cost = float(KernelCost(k, network).total_cost())
        options.append(
            MatmulLayoutOption(
                output_layout_str=out_str,
                operand_a_layout_str=a_str,
                operand_b_layout_str=b_str,
                total_cost_lan=cost,
                rolls_flag_used=roll_flag,
                source=source,
            )
        )
        if collect_kernel_refs:
            kernel_refs.append(k)

    for base in base_kernels:
        add_from_kernel(base, "enumerate")

    attempts_global = 0
    stop_all = False
    for base in base_kernels:
        if stop_all:
            break
        seed_trials = 0
        skip_base = False
        a0, b0 = base.cs[0], base.cs[1]
        for side, op_seed in (("A", a0), ("B", b0)):
            if stop_all or skip_base:
                break
            for op_var in _operand_neighborhood_kernels(op_seed, mode=neighbor_mode):
                if max_local_attempts is not None and attempts_global >= max_local_attempts:
                    stop_all = True
                    break
                if max_attempts_per_seed is not None and seed_trials >= max_attempts_per_seed:
                    skip_base = True
                    break
                attempts_global += 1
                seed_trials += 1
                if side == "A":
                    pair = (op_var, b0)
                else:
                    pair = (a0, op_var)
                if not check_alignment(term, alignment, pair[0], pair[1]):
                    continue
                try:
                    out_k = output_layout(term, alignment, pair[0], pair[1])
                except Exception:
                    continue
                if out_k.op != KernelOp.MATMUL:
                    continue
                add_from_kernel(out_k, f"local_{side}")

    if collect_kernel_refs:
        paired = list(zip(options, kernel_refs, strict=True))
        paired.sort(key=lambda t: (t[0].total_cost_lan, t[0].output_layout_str))
        return [p[0] for p in paired], [p[1] for p in paired]
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
    p.add_argument(
        "--local-search",
        action="store_true",
        help="after enumeration, search operand permute/roll neighbors + output_layout",
    )
    p.add_argument(
        "--max-local-attempts",
        type=int,
        default=800,
        help="cap neighbor trials for --local-search (0 = unlimited)",
    )
    p.add_argument(
        "--neighbor-mode",
        choices=("default", "wide"),
        default="default",
        help="wide = all dim permutations when len(dims)≤6 (larger neighbor set)",
    )
    p.add_argument(
        "--max-attempts-per-seed",
        type=int,
        default=0,
        help="cap neighbor trials per base MATMUL kernel (0 = unlimited; --local-search only)",
    )
    p.add_argument(
        "--verify-lower",
        action="store_true",
        help="for --local-search, run Lower() on top kernels (requires extra kernel refs pass)",
    )
    args = p.parse_args(argv)

    roll_flag = not args.no_rolls
    if args.verify_lower and not args.local_search:
        p.error("--verify-lower requires --local-search")
    if args.dedupe and args.verify_lower:
        p.error("--dedupe cannot be combined with --verify-lower (kernel refs are dropped)")
    if args.local_search:
        max_att = args.max_local_attempts if args.max_local_attempts > 0 else None
        max_per_seed = args.max_attempts_per_seed if args.max_attempts_per_seed > 0 else None
        if args.verify_lower:
            opts, kerns = local_search_matmul_kernels(
                m=args.m,
                k=args.k,
                n_out=args.n_out,
                n_slots=args.slots,
                roll_flag=roll_flag,
                max_local_attempts=max_att,
                neighbor_mode=args.neighbor_mode,
                max_attempts_per_seed=max_per_seed,
                collect_kernel_refs=True,
            )
        else:
            opts = local_search_matmul_kernels(
                m=args.m,
                k=args.k,
                n_out=args.n_out,
                n_slots=args.slots,
                roll_flag=roll_flag,
                max_local_attempts=max_att,
                neighbor_mode=args.neighbor_mode,
                max_attempts_per_seed=max_per_seed,
            )
            kerns = None
    else:
        opts = explore_matmul_kernels(
            m=args.m,
            k=args.k,
            n_out=args.n_out,
            n_slots=args.slots,
            roll_flag=roll_flag,
        )
        kerns = None
    if args.dedupe:
        opts = dedupe_matmul_options(opts)
        kerns = None

    lower_rows: list[dict[str, Any]] = []
    if args.verify_lower and kerns is not None:
        for o, ker in zip(opts[: args.top], kerns[: args.top], strict=False):
            lr = try_lower_matmul_kernel(ker)
            lower_rows.append(
                {
                    "cost": o.total_cost_lan,
                    "source": o.source,
                    "lower_ok": lr["ok"],
                    "lower_error": lr.get("error"),
                }
            )

    if args.json:
        rows = [o.to_json_dict() for o in opts[: args.top]]
        if lower_rows:
            for row, extra in zip(rows, lower_rows, strict=True):
                row["lower_ok"] = extra["lower_ok"]
                if extra.get("lower_error"):
                    row["lower_error"] = extra["lower_error"]
        print(json.dumps(rows, indent=2))
        return

    print("MATMUL alignment (2D @ 2D):", sorted_alignment_pairs_2d2d())
    extra = []
    if args.local_search:
        extra.append(f"neighbor_mode={args.neighbor_mode}")
        if args.max_attempts_per_seed:
            extra.append(f"per_seed_cap={args.max_attempts_per_seed}")
    print(
        f"Shapes (logical) A=[{args.m},{args.k}] B=[{args.k},{args.n_out}] "
        f"n_slots={args.slots} roll_flag={roll_flag}"
        + (f" local_search max_attempts={args.max_local_attempts}" if args.local_search else "")
        + (f" ({', '.join(extra)})" if extra else "")
    )
    print(f"Found {len(opts)} kernels (showing up to {args.top} cheapest by LAN cost)\n")
    for i, o in enumerate(opts[: args.top], 1):
        tag = f" [{o.source}]" if o.source != "enumerate" else ""
        low = ""
        if lower_rows and i <= len(lower_rows):
            lr = lower_rows[i - 1]
            low = f" lower={'ok' if lr['lower_ok'] else 'FAIL'}"
        print(f"--- {i} cost={o.total_cost_lan:.4f}{tag}{low} ---")
        print(f"  A: {o.operand_a_layout_str}")
        print(f"  B: {o.operand_b_layout_str}")
        print(f"  out: {o.output_layout_str}")


if __name__ == "__main__":
    main()
