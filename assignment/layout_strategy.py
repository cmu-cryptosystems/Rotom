"""
Evolvable layout-assignment strategy (baseline matches legacy Rotom behavior).

OpenEvolve mutates this file (or a copy pointed at by ROTOM_LAYOUT_STRATEGY_PATH).
Stable API (all entry points must exist):
  - pre_optimizer_kernels(kernels, term, network)
  - optimizer_pass_config(roll_flag)
  - layout_sort_key(kernel, network)
  - postprocess_kernels(kernels, term, network)
  - cost_model_multipliers(network)
"""

from __future__ import annotations

from typing import Any


def pre_optimizer_kernels(kernels: list, term: Any, network: str) -> list:
    """
    Hook on raw candidates from codegen, before roll / BSGS optimizer passes.

    Baseline: identity. Evolved code may reorder or drop candidates (dropping risks
    missing cheaper layouts; Toy checks still enforce correctness for survivors).
    """
    return list(kernels)


def optimizer_pass_config(roll_flag: bool) -> dict[str, bool]:
    """
    Toggle individual roll-optimization passes when roll_flag is True.

    When roll_flag is False, Rotom skips this pipeline entirely (see Optimizer.run).
    Keys are ignored if roll_flag is False.

    Passes correspond to the sequence in opt/opt.py (legacy order).
    """
    if not roll_flag:
        return {}
    return {
        "roll_propagation": True,
        "roll_reordering": True,
        "rot_roll": True,
        "ct_roll_bsgs": True,
        "bsgs_matmul": True,
    }


def layout_sort_key(kernel: Any, network: str) -> str:
    """
    Deterministic sort key for kernels after Optimizer.run (must be total-order friendly).

    Baseline: legacy Rotom uses layout_str(). Evolved strategies may prepend a rank
    prefix (e.g. f"{score:016.8f}|{kernel.layout.layout_str()}") to bias tie handling
    while staying deterministic.
    """
    return kernel.layout.layout_str()


def postprocess_kernels(kernels: list, term: Any, network: str) -> list:
    """
    Optional hook after shape_check and equivalent-kernel expansion, before cost minimization.

    Baseline: return kernels unchanged (same multiset as legacy Rotom).

    Evolved code may reorder or subsample candidates. Subsampling can reduce cost
    at the risk of missing the global minimum over Rotom's discrete search space.
    """
    return list(kernels)


def cost_model_multipliers(network: str) -> dict[str, float]:
    """
    Per-operation multipliers applied to Rotom's analytic cost model during *search*
    (which layout wins), not to the evaluator's reported baseline cost.

    Return {} for baseline (no change). Keys are subset of comm, add, mul, rot.
    Example: {"rot": 1.5, "mul": 0.9} makes the compiler prefer fewer rotations
    during assignment; evaluator still measures realized cost with the default model.
    """
    return {}
