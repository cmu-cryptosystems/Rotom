"""
OpenEvolve mutates **search hyperparameters** for ``matmul_layout_explore.local_search``.

The evaluator compares the best ``KernelCost`` from local search (with your settings)
against a fixed enumeration baseline on the same shape. Tune budgets and neighbor
breadth; invalid combinations (unknown mode, negative ints) fail evaluation.

Do not rename ``get_matmul_search_config``. Keep imports minimal.
"""

from __future__ import annotations

from typing import Any


def get_matmul_search_config() -> dict[str, Any]:
    """
    Keys (all optional beyond these defaults):

    - ``neighbor_mode``: ``\"default\"`` (cyclic dim perms + roll pairs) or
      ``\"wide\"`` (all traversal permutations when dim count ≤ 6, else like default).
    - ``max_local_attempts``: int or ``0`` for unlimited (cap total neighbor *trials*).
    - ``max_attempts_per_seed``: int or ``0`` for unlimited (per base MATMUL kernel).
    - ``m``, ``k``, ``n``, ``slots``: problem shape and HE slot count for the eval instance.
    - ``roll_flag``: bool — should match whether you expect sum-roll paths.
    """
    return {
        "neighbor_mode": "default",
        "max_local_attempts": 4000,
        "max_attempts_per_seed": 0,
        "m": 4,
        "k": 4,
        "n": 4,
        "slots": 256,
        "roll_flag": False,
    }
