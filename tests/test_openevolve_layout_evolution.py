"""
Tests for OpenEvolve-driven layout assignment (bench_worker + evaluator).

Fast runs use ROTOM_EVOLVE_BENCHES=matmul_128_64 (~30s per full evaluate()).
Full tensor-program coverage: pytest -m "openevolve and slow".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "evolve_openevolve" / "bench_worker.py"
BASELINE_STRATEGY = ROOT / "assignment" / "layout_strategy.py"


def _run_worker(
    strategy_path: Path,
    *,
    benches: str | None = "matmul_128_64",
    timeout: int = 300,
) -> tuple[int, dict]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if benches is not None:
        env["ROTOM_EVOLVE_BENCHES"] = benches
    else:
        env.pop("ROTOM_EVOLVE_BENCHES", None)
    proc = subprocess.run(
        [sys.executable, str(WORKER), str(strategy_path)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    raw = proc.stdout.strip().splitlines()
    try:
        data = json.loads(raw[-1]) if raw else {}
    except json.JSONDecodeError:
        data = {"parse_error": True, "stdout": proc.stdout, "stderr": proc.stderr}
    return proc.returncode, data


WORSE_STRATEGY_SOURCE = '''
from __future__ import annotations
from typing import Any

# Same API as assignment.layout_strategy; disables BSGS matmul pass (often hurts cost).

def pre_optimizer_kernels(kernels: list, term: Any, network: str) -> list:
    return list(kernels)

def optimizer_pass_config(roll_flag: bool) -> dict[str, bool]:
    if not roll_flag:
        return {}
    return {
        "roll_propagation": True,
        "roll_reordering": True,
        "rot_roll": True,
        "ct_roll_bsgs": True,
        "bsgs_matmul": False,
    }

def layout_sort_key(kernel: Any, network: str) -> str:
    return kernel.layout.layout_str()

def postprocess_kernels(kernels: list, term: Any, network: str) -> list:
    return list(kernels)

def cost_model_multipliers(network: str) -> dict[str, float]:
    return {}
'''


@pytest.fixture
def fast_benches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROTOM_EVOLVE_BENCHES", "matmul_128_64")


@pytest.fixture
def two_benches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ROTOM_EVOLVE_BENCHES", "matmul_128_64,double_matmul_128_64"
    )


@pytest.mark.openevolve
def test_bench_worker_baseline_fast():
    code, data = _run_worker(BASELINE_STRATEGY, benches="matmul_128_64")
    assert code == 0, data
    assert data.get("ok") is True
    assert "matmul_128_64" in data.get("benchmarks", {})
    mm = data["benchmarks"]["matmul_128_64"]
    assert mm["semantic_ok"] is True
    assert mm["total_cost"] > 0


@pytest.mark.openevolve
def test_bench_worker_rejects_unknown_benchmark_name():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["ROTOM_EVOLVE_BENCHES"] = "not_a_benchmark"
    proc = subprocess.run(
        [sys.executable, str(WORKER), str(BASELINE_STRATEGY)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2
    tail = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "unknown" in (tail.get("error") or "").lower()


@pytest.mark.openevolve
def test_evaluator_same_as_baseline_is_one(fast_benches, tmp_path: Path):
    from evolve_openevolve.evaluator import evaluate

    twin = tmp_path / "layout_strategy_twin.py"
    twin.write_text(BASELINE_STRATEGY.read_text(encoding="utf-8"), encoding="utf-8")

    r = evaluate(str(twin))
    assert r.metrics["combined_score"] == pytest.approx(1.0, rel=0, abs=1e-6)
    assert r.metrics["num_benchmarks"] == 1.0


@pytest.mark.openevolve
def test_evaluator_worse_strategy_scores_below_baseline(fast_benches, tmp_path: Path):
    from evolve_openevolve.evaluator import evaluate

    worse = tmp_path / "worse.py"
    worse.write_text(WORSE_STRATEGY_SOURCE, encoding="utf-8")

    r = evaluate(str(worse))
    assert r.metrics["combined_score"] < 1.0
    assert r.metrics.get("semantic_failure", 0) != 1.0
    assert r.metrics["num_benchmarks"] == 1.0


@pytest.mark.openevolve
@pytest.mark.slow
def test_evaluator_two_tensor_programs(two_benches, tmp_path: Path):
    """Two benchmarks: matmul chain + double matmul (still ~1–2 min)."""
    from evolve_openevolve.evaluator import evaluate

    twin = tmp_path / "layout_strategy_twin.py"
    twin.write_text(BASELINE_STRATEGY.read_text(encoding="utf-8"), encoding="utf-8")

    r = evaluate(str(twin))
    assert r.metrics["combined_score"] == pytest.approx(1.0, rel=0, abs=1e-6)
    assert r.metrics["num_benchmarks"] == 2.0


@pytest.mark.openevolve
@pytest.mark.slow
def test_bench_worker_all_default_programs():
    """All three packaged tensor programs (several minutes)."""
    code, data = _run_worker(BASELINE_STRATEGY, benches=None, timeout=600)
    assert code == 0, data
    assert data.get("ok") is True
    names = set(data["benchmarks"])
    assert names == {
        "matmul_128_64",
        "matmul_256_128",
        "double_matmul_128_64",
    }
    for meta in data["benchmarks"].values():
        assert meta["semantic_ok"] is True
