"""
OpenEvolve evaluator: semantic check vs plaintext reference, score vs baseline strategy cost.

Run from repo root (evolve/) or Rotom parent with:
  openevolve-run Rotom/assignment/layout_strategy.py Rotom/evolve_openevolve/evaluator.py \\
    --config Rotom/evolve_openevolve/config.yaml --iterations N

Put API keys in Rotom/.env (gitignored), e.g. OPENAI_API_KEY=... .
OpenEvolve's CLI loads .env from cwd, config dir, and Rotom roots before
creating the LLM client.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # pragma: no cover — full OpenEvolve install provides this
    from dataclasses import dataclass, field
    from typing import Dict, Union

    @dataclass
    class EvaluationResult:
        metrics: Dict[str, float]
        artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)

        def has_artifacts(self) -> bool:
            return bool(self.artifacts)


def _rotom_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_worker(strategy_path: str, timeout_sec: int = 600) -> dict:
    root = _rotom_root()
    worker = root / "evolve_openevolve" / "bench_worker.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    proc = subprocess.run(
        [sys.executable, str(worker), strategy_path],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        return {
            "ok": False,
            "benchmarks": {},
            "error": proc.stderr or f"exit {proc.returncode}",
        }
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {
            "ok": False,
            "benchmarks": {},
            "error": f"bad worker output: stderr={proc.stderr!r} stdout={proc.stdout!r}",
        }


def evaluate(program_path: str):
    """
    Evaluate evolved layout_strategy module at program_path.

    Baseline costs come from the committed Rotom assignment/layout_strategy.py
    (default hooks), not from the evolved file.

    If ``ROTOM_EVOLVE_BENCHES`` is set in the environment (comma-separated names:
    matmul_128_64, matmul_256_128, double_matmul_128_64), the worker runs only
    that subset (used by pytest for faster checks).
    """
    rotom = _rotom_root()
    baseline_strategy = rotom / "assignment" / "layout_strategy.py"

    baseline = _run_worker(str(baseline_strategy))
    evolved = _run_worker(os.path.abspath(program_path))

    artifacts: dict[str, str] = {
        "baseline_json": json.dumps(baseline, indent=2),
        "evolved_json": json.dumps(evolved, indent=2),
    }

    if not baseline.get("ok"):
        artifacts["note"] = "baseline run failed; check Rotom install and benchmarks"
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "error": 1.0,
            },
            artifacts=artifacts,
        )

    if evolved.get("error"):
        artifacts["failure"] = str(evolved["error"])

    bbench = baseline.get("benchmarks") or {}
    ebench = evolved.get("benchmarks") or {}

    if not evolved.get("ok"):
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "semantic_failure": 1.0,
            },
            artifacts=artifacts,
        )

    ratios = []
    cost_deltas = []
    for name, bmeta in bbench.items():
        if name not in ebench:
            continue
        emeta = ebench[name]
        if not emeta.get("semantic_ok"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "semantic_failure": 1.0},
                artifacts=artifacts,
            )
        bc = float(bmeta.get("total_cost", 0.0))
        ec = max(float(emeta.get("total_cost", 0.0)), 1e-12)
        ratios.append(bc / ec)
        cost_deltas.append(bc - ec)

    if not ratios:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts=artifacts,
        )

    combined_score = float(sum(ratios) / len(ratios))
    metrics = {
        "combined_score": combined_score,
        "mean_baseline_over_evolved_cost": combined_score,
        "mean_cost_delta_vs_baseline": float(sum(cost_deltas) / len(cost_deltas)),
        "num_benchmarks": float(len(ratios)),
    }
    artifacts["summary"] = (
        f"mean(baseline_cost/evolved_cost)={combined_score:.6f}; "
        f"per-bench ratios={[round(r, 4) for r in ratios]}"
    )
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
