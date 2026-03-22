"""SiLU activation statistics and polynomial SiLU for the plaintext ResNet20.

- **Calibration**: For each of the 19 SiLU sites, record per-batch min/max of inputs,
  then report the *average* batch low / average batch high (mean of batch minima /
  mean of batch maxima). Also records global min/max across calibration for reference.
- **Poly fit**: Least-squares polynomial for SiLU on ``[lo, hi]`` (ascending coeffs),
  matching ``tests/test_silu_poly.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _reference_silu_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def _chebyshev_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    j = np.arange(n, dtype=np.float64) + 0.5
    t = np.cos(np.pi * j / n)
    return 0.5 * (hi - lo) * t + 0.5 * (hi + lo)


def silu_poly_ascending_coeffs(
    lo: float, hi: float, degree: int, n_nodes: int = 80
) -> List[float]:
    """Least-squares polynomial for SiLU on [lo, hi]; coeffs for c0 + c1*x + ... + c_d*x^d."""
    xs = _chebyshev_nodes(lo, hi, n_nodes)
    ys = _reference_silu_np(xs)
    high_to_low = np.polyfit(xs, ys, degree)
    return [float(c) for c in high_to_low[::-1]]


def resnet20_silu_site_names() -> List[str]:
    """Human-readable names for the 19 SiLU inputs in forward order."""
    names = ["stem_conv1_bn"]
    for layer in ("layer1", "layer2", "layer3"):
        for bi in range(3):
            names.append(f"{layer}_block{bi}_post_conv1_bn")
            names.append(f"{layer}_block{bi}_post_add")
    return names


def iter_resnet20_silu_modules(model: nn.Module) -> Iterator[nn.Module]:
    """Forward order of SiLU modules in `resnet20` (1 + 9 blocks * 2 = 19)."""
    if not hasattr(model, "act_conv1"):
        raise ValueError("expected ResNet with act_conv1 (see resnet_model.resnet20)")
    yield model.act_conv1
    for layer in (model.layer1, model.layer2, model.layer3):
        for block in layer:
            yield block.act1
            yield block.act2


@dataclass
class SiluBounds:
    """Per-SiLU statistics over a calibration run."""

    avg_low: float
    avg_high: float
    global_min: float
    global_max: float


def format_silu_bounds_table(bounds: Sequence[SiluBounds]) -> str:
    """Printable table: per-site average batch low/high and global min/max."""
    names = resnet20_silu_site_names()
    lines = [
        f"{'site':<32} {'avg_low':>12} {'avg_high':>12} {'g_min':>12} {'g_max':>12}"
    ]
    for i, b in enumerate(bounds):
        label = names[i] if i < len(names) else f"site_{i}"
        lines.append(
            f"{label:<32} {b.avg_low:12.6f} {b.avg_high:12.6f} "
            f"{b.global_min:12.6f} {b.global_max:12.6f}"
        )
    return "\n".join(lines)


def calibrate_silu_input_bounds(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int | None = 32,
) -> Tuple[List[SiluBounds], int]:
    """
    Run the model on ``loader`` and aggregate bounds at each SiLU input.

    For each SiLU site we record, per batch, the min and max of the activation input.
    **avg_low** / **avg_high** are the mean of those batch minima and batch maxima.

    Returns:
        (bounds_per_site, num_batches_used)
    """
    silu_list = list(iter_resnet20_silu_modules(model))
    n = len(silu_list)
    batch_mins: List[List[float]] = [[] for _ in range(n)]
    batch_maxs: List[List[float]] = [[] for _ in range(n)]
    global_min = [float("inf")] * n
    global_max = [float("-inf")] * n

    hooks = []

    def make_hook(idx: int):
        def hook(_module, inputs, _output):
            x = inputs[0]
            xv = x.detach().float()
            bmin = float(xv.min().cpu())
            bmax = float(xv.max().cpu())
            batch_mins[idx].append(bmin)
            batch_maxs[idx].append(bmax)
            global_min[idx] = min(global_min[idx], bmin)
            global_max[idx] = max(global_max[idx], bmax)

        return hook

    for i, m in enumerate(silu_list):
        hooks.append(m.register_forward_hook(make_hook(i)))

    model.eval()
    batches = 0
    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device)
            model(images)
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    for h in hooks:
        h.remove()

    out: List[SiluBounds] = []
    for i in range(n):
        lows = batch_mins[i]
        highs = batch_maxs[i]
        if not lows:
            raise RuntimeError(f"No batches captured for SiLU site {i}")
        avg_low = float(np.mean(lows))
        avg_high = float(np.mean(highs))
        out.append(
            SiluBounds(
                avg_low=avg_low,
                avg_high=avg_high,
                global_min=global_min[i],
                global_max=global_max[i],
            )
        )
    return out, batches


class PolySiLU(nn.Module):
    """SiLU replaced by a fixed polynomial ``sum_i c_i x^i`` (ascending powers)."""

    def __init__(self, coeffs: Sequence[float]):
        super().__init__()
        t = torch.as_tensor(list(coeffs), dtype=torch.float32)
        self.register_buffer("coeffs", t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.coeffs
        if c.numel() == 0:
            return x
        out = torch.zeros_like(x, dtype=torch.float32)
        xp = torch.ones_like(x, dtype=torch.float32)
        x32 = x.to(dtype=torch.float32)
        for i in range(c.numel()):
            out = out + c[i] * xp
            if i + 1 < c.numel():
                xp = xp * x32
        return out.to(dtype=x.dtype)


def replace_resnet20_silu_with_poly(
    model: nn.Module,
    bounds: Sequence[SiluBounds],
    degree: int = 11,
    n_nodes: int = 80,
) -> None:
    """
    In-place: replace each ``nn.SiLU`` with ``PolySiLU`` using ``[avg_low, avg_high]``
    from calibration (clamped so lo < hi).
    """
    n_expected = len(list(iter_resnet20_silu_modules(model)))
    if n_expected != len(bounds):
        raise ValueError(f"expected {n_expected} bound records, got {len(bounds)}")

    polys: List[PolySiLU] = []
    for b in bounds:
        lo, hi = b.avg_low, b.avg_high
        if lo >= hi:
            lo, hi = b.global_min, b.global_max
        if lo >= hi:
            lo, hi = -1.0, 1.0
        coeffs = silu_poly_ascending_coeffs(lo, hi, degree, n_nodes=n_nodes)
        polys.append(PolySiLU(coeffs))

    model.act_conv1 = polys[0]
    k = 1
    for layer in (model.layer1, model.layer2, model.layer3):
        for block in layer:
            block.act1 = polys[k]
            k += 1
            block.act2 = polys[k]
            k += 1
    assert k == len(polys)
