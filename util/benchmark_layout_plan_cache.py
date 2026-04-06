"""On-disk cache for expensive ``apply_layout`` precomputations (layout plans).

Plans are keyed by layout geometry, ``ndim``, ``layout_len``, and ``n`` (see
``layout_util._apply_layout_plan_cache_key``). Files are shared across runs when the
key matches.

Typical layout::

    <repo>/.cache/layout_plans/<benchmark>/n_<ring_dim>/<sha256>.pkl

This module optionally overrides the cache directory on a per-benchmark basis.
When not overridden, ``ROTOM_APPLY_LAYOUT_PLAN_CACHE_DIR`` (if set) is still
respected by ``layout_util``.

Enable/disable via ``--cache`` / ``--no-cache`` on ``main.py`` / ``e2e.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

from util.layout_util import set_apply_layout_plan_disk_cache_dir

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_ROOT = _REPO_ROOT / ".cache" / "layout_plans"
_MAX_KEY_LEN = 200


def _sanitize_benchmark_key(name: str) -> str:
    s = (name or "").strip() or "unknown"
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s[:_MAX_KEY_LEN]


def benchmark_layout_plan_cache_path(benchmark_key: str, n: int | None = None) -> Path:
    """Directory where pickle plans for this benchmark (and optional ``n``) are stored."""
    bench = _sanitize_benchmark_key(benchmark_key)
    p = _CACHE_ROOT / bench
    if n is not None:
        p = p / f"n_{int(n)}"
    return p


def activate_benchmark_layout_plan_cache(
    benchmark_key: str, n: int | None = None
) -> Path:
    """Create the cache directory and route layout-plan disk I/O there."""
    out = benchmark_layout_plan_cache_path(benchmark_key, n=n)
    out.mkdir(parents=True, exist_ok=True)
    set_apply_layout_plan_disk_cache_dir(str(out))
    return out


def clear_benchmark_layout_plan_cache_override() -> None:
    """Stop using the per-benchmark directory (fall back to env only)."""
    set_apply_layout_plan_disk_cache_dir(None)


def maybe_install_layout_plan_cache_from_args(args, benchmark_key: str) -> None:
    """Optionally override the layout-plan cache directory based on CLI args.

    If caching is disabled, or if ``benchmark_key`` is empty/``main``, the override
    is cleared and ``layout_util`` falls back to environment configuration only.
    """
    cache_enabled = bool(getattr(args, "cache", False))
    key = (benchmark_key or "").strip()

    if not cache_enabled or not key or key == "main":
        clear_benchmark_layout_plan_cache_override()
        return

    activate_benchmark_layout_plan_cache(key, getattr(args, "n", None))
