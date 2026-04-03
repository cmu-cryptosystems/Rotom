"""
OpenEvolve evaluator: evolve ``matmul_layout_program.get_matmul_search_config``.

Loads the evolved program, runs ``local_search_matmul_kernels`` with returned knobs,
and scores **baseline enumeration min cost / evolved min cost** (higher is better when
the search finds cheaper layouts).

Run (from ``Rotom/`` with ``PYTHONPATH=`` and OpenEvolve on the path)::

  python -m openevolve.cli \\
    evolve_openevolve/matmul_layout_program.py \\
    evolve_openevolve/matmul_layout_evaluator.py \\
    -c evolve_openevolve/matmul_layout_config.yaml
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # pragma: no cover
    from dataclasses import dataclass, field
    from typing import Dict, Union

    @dataclass
    class EvaluationResult:
        metrics: Dict[str, float]
        artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)

        def has_artifacts(self) -> bool:
            return bool(self.artifacts)


def _norm_attempt_cap(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return None if n <= 0 else n


def _load_search_config(program_path: str) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("matmul_layout_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "get_matmul_search_config", None)
    if not callable(fn):
        raise AttributeError("program must define get_matmul_search_config() -> dict")
    cfg = fn()
    if not isinstance(cfg, dict):
        raise TypeError("get_matmul_search_config must return a dict")
    return cfg


def evaluate(program_path: str):
    from evolve_openevolve.matmul_layout_explore import (
        explore_matmul_kernels,
        local_search_matmul_kernels,
    )

    try:
        raw = _load_search_config(program_path)
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts={"failure": f"load config: {type(e).__name__}: {e}"},
        )

    nm = raw.get("neighbor_mode", "default")
    if nm not in ("default", "wide"):
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts={"failure": f"invalid neighbor_mode {nm!r}"},
        )

    m = int(raw.get("m", 4))
    k_dim = int(raw.get("k", 4))
    n_out = int(raw.get("n", 4))
    slots = int(raw.get("slots", 256))
    roll_flag = bool(raw.get("roll_flag", False))
    max_local = _norm_attempt_cap(raw.get("max_local_attempts", 4000))
    max_per_seed = _norm_attempt_cap(raw.get("max_attempts_per_seed", 0))

    if m <= 0 or k_dim <= 0 or n_out <= 0 or slots <= 0:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts={"failure": "m, k, n, slots must be positive"},
        )

    try:
        baseline = explore_matmul_kernels(
            m=m,
            k=k_dim,
            n_out=n_out,
            n_slots=slots,
            roll_flag=roll_flag,
            network="lan",
        )
        evolved = local_search_matmul_kernels(
            m=m,
            k=k_dim,
            n_out=n_out,
            n_slots=slots,
            roll_flag=roll_flag,
            network="lan",
            max_local_attempts=max_local,
            neighbor_mode=nm,
            max_attempts_per_seed=max_per_seed,
        )
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts={"failure": f"search: {type(e).__name__}: {e}"},
        )

    if not baseline or not evolved:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts={"failure": "empty baseline or evolved result"},
        )

    min_b = min(o.total_cost_lan for o in baseline)
    min_e = min(o.total_cost_lan for o in evolved)
    denom = max(min_e, 1e-12)
    ratio = float(min_b / denom)

    summary = {
        "m": m,
        "k": k_dim,
        "n": n_out,
        "slots": slots,
        "roll_flag": roll_flag,
        "neighbor_mode": nm,
        "max_local_attempts": max_local,
        "max_attempts_per_seed": max_per_seed,
        "baseline_min_cost": min_b,
        "evolved_min_cost": min_e,
        "count_baseline": len(baseline),
        "count_evolved": len(evolved),
        "cost_ratio_baseline_over_evolved": ratio,
    }

    return EvaluationResult(
        metrics={
            "combined_score": ratio,
            "baseline_min_cost": float(min_b),
            "evolved_min_cost": float(min_e),
        },
        artifacts={"summary": json.dumps(summary, indent=2)},
    )
