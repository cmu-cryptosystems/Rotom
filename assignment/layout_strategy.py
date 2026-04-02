"""
Evolvable layout-assignment strategy (baseline matches legacy Rotom behavior).

OpenEvolve mutates this file (or a copy pointed at by ROTOM_LAYOUT_STRATEGY_PATH).
The public API must remain stable:
  - optimizer_pass_config(roll_flag)
  - postprocess_kernels(kernels, term, network)
"""

from __future__ import annotations

from typing import Any


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


def postprocess_kernels(kernels: list, term: Any, network: str) -> list:
    """
    Optional hook after shape_check and equivalent-kernel expansion, before cost minimization.

    Baseline: return kernels unchanged (same multiset as legacy Rotom).

    Evolved code may reorder or subsample candidates. Subsampling can reduce cost
    at the risk of missing the global minimum over Rotom's discrete search space.
    """
    return list(kernels)
