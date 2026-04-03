"""
OpenEvolve evaluator: evolve ``layout_transform_program.propose_transformed_kernel``.

Run (from ``Rotom/`` with PYTHONPATH including OpenEvolve if needed)::

  python -m openevolve.cli \\
    evolve_openevolve/layout_transform_program.py \\
    evolve_openevolve/layout_transform_evaluator.py \\
    -c evolve_openevolve/layout_transform_config.yaml
"""

from __future__ import annotations

import json

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


def evaluate(program_path: str):
    from evolve_openevolve.layout_transform_search import (
        baseline_program_path,
        build_seed_tensor_kernel,
        evaluate_transformed_kernel,
        load_propose_fn,
    )

    term, seed, shape = build_seed_tensor_kernel()
    baseline_path = str(baseline_program_path())

    try:
        propose_baseline = load_propose_fn(baseline_path)
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts={"failure": f"baseline program load: {e}"},
        )

    try:
        propose_evolved = load_propose_fn(program_path)
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts={"failure": f"evolved program load: {e}"},
        )

    b = evaluate_transformed_kernel(term, seed, shape, propose_baseline)
    e = evaluate_transformed_kernel(term, seed, shape, propose_evolved)

    artifacts = {
        "baseline_eval": json.dumps(b, indent=2),
        "evolved_eval": json.dumps(e, indent=2),
    }

    if not b.get("ok"):
        artifacts["note"] = "baseline TENSOR transform pipeline failed"
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts=artifacts,
        )

    if not e.get("ok") or not e.get("semantic_ok"):
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "semantic_failure": 1.0,
            },
            artifacts=artifacts,
        )

    bc = max(float(b["total_cost"]), 1e-12)
    ec = max(float(e["total_cost"]), 1e-12)
    ratio = bc / ec
    artifacts["summary"] = (
        f"baseline_cost={bc:.6f} evolved_cost={ec:.6f} ratio={ratio:.6f}; "
        f"layouts {b.get('layout_str_before')} -> {e.get('layout_str_after')}"
    )

    return EvaluationResult(
        metrics={
            "combined_score": float(ratio),
            "baseline_cost": float(bc),
            "evolved_cost": float(ec),
        },
        artifacts=artifacts,
    )
