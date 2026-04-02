"""Load layout strategy from ROTOM_LAYOUT_STRATEGY_PATH or default package module."""

from __future__ import annotations

import importlib.util
import os
from types import ModuleType


def get_strategy_module() -> ModuleType:
    path = os.environ.get("ROTOM_LAYOUT_STRATEGY_PATH")
    if path:
        abs_path = os.path.abspath(path)
        spec = importlib.util.spec_from_file_location(
            "rotom_layout_strategy_dynamic", abs_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load strategy from {abs_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    from assignment import layout_strategy

    return layout_strategy
