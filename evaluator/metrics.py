"""
Placement quality metrics.

All functions take a placement result dict and return scalar floats.
This file is FIXED — not modified during experiments.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_hpwl(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    net_cell_indices: List[List[int]],
    net_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Half-perimeter wirelength (HPWL) across all nets.

    HPWL(net) = (max_x - min_x) + (max_y - min_y) over all pins in net.
    Total HPWL = sum over all nets.

    Args:
        cell_x: (N,) array of cell x-coordinates
        cell_y: (N,) array of cell y-coordinates
        net_cell_indices: list of lists; net_cell_indices[i] = cells in net i
        net_weights: (M,) optional per-net weight; default all-ones

    Returns:
        Total HPWL (float)
    """
    total = 0.0
    for i, cells in enumerate(net_cell_indices):
        if len(cells) < 2:
            continue
        xs = cell_x[cells]
        ys = cell_y[cells]
        hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
        weight = 1.0 if net_weights is None else float(net_weights[i])
        total += weight * hpwl
    return total


def compute_overflow(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_w: np.ndarray,
    cell_h: np.ndarray,
    die_w: float,
    die_h: float,
    num_bins_x: int = 512,
    num_bins_y: int = 512,
    target_density: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute bin-wise density overflow.

    Returns:
        (mean_overflow, top5_overflow)
        where overflow_k = max(0, density_k - target_density)
        and top5_overflow is the mean over the 5 highest-overflow bins.
    """
    bin_w = die_w / num_bins_x
    bin_h = die_h / num_bins_y
    density = np.zeros((num_bins_x, num_bins_y), dtype=np.float64)

    for i in range(len(cell_x)):
        cx, cy = cell_x[i], cell_y[i]
        cw, ch = cell_w[i], cell_h[i]

        # Cell bin range (integer)
        bx_lo = max(0, int((cx - cw / 2) / bin_w))
        bx_hi = min(num_bins_x - 1, int((cx + cw / 2) / bin_w))
        by_lo = max(0, int((cy - ch / 2) / bin_h))
        by_hi = min(num_bins_y - 1, int((cy + ch / 2) / bin_h))

        cell_area = cw * ch
        num_bins = max(1, (bx_hi - bx_lo + 1) * (by_hi - by_lo + 1))
        area_per_bin = cell_area / num_bins

        density[bx_lo : bx_hi + 1, by_lo : by_hi + 1] += area_per_bin / (bin_w * bin_h)

    overflow = np.maximum(0.0, density - target_density)
    mean_ovfl = float(overflow.mean())
    flat = overflow.flatten()
    top5_ovfl = float(np.sort(flat)[-5:].mean()) if len(flat) >= 5 else mean_ovfl
    return mean_ovfl, top5_ovfl


def compute_tns_proxy(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    net_cell_indices: List[List[int]],
    critical_net_flags: Optional[np.ndarray] = None,
    timing_slack_threshold: float = 0.0,
) -> float:
    """
    Lightweight TNS proxy: sum of excess wirelength on critical-path nets.

    True TNS requires a full router and STA tool. This proxy estimates it
    by computing how much the critical-path net lengths exceed a threshold.
    Replace with full STA when hardware is available.

    Args:
        cell_x, cell_y: cell coordinates
        net_cell_indices: connectivity
        critical_net_flags: (M,) bool array; which nets are on timing-critical paths
        timing_slack_threshold: HPWL threshold above which slack becomes negative

    Returns:
        TNS proxy (lower is better; 0 = no timing violation)
    """
    if critical_net_flags is None:
        # Without timing info, use nets with high fanout as proxy for criticality
        critical_net_flags = np.array(
            [len(cells) >= 4 for cells in net_cell_indices], dtype=bool
        )

    tns = 0.0
    for i, cells in enumerate(net_cell_indices):
        if not critical_net_flags[i] or len(cells) < 2:
            continue
        xs = cell_x[cells]
        ys = cell_y[cells]
        hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
        # Excess above threshold = negative slack contribution
        excess = max(0.0, hpwl - timing_slack_threshold)
        tns += excess

    return tns


def normalize_hpwl(hpwl: float, baseline_hpwl: float) -> float:
    """Return HPWL normalized to baseline (1.0 = matches baseline, <1.0 = better)."""
    return hpwl / baseline_hpwl if baseline_hpwl > 0 else float("inf")


def compute_all_metrics(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_w: np.ndarray,
    cell_h: np.ndarray,
    net_cell_indices: List[List[int]],
    die_w: float,
    die_h: float,
    net_weights: Optional[np.ndarray] = None,
    critical_net_flags: Optional[np.ndarray] = None,
    baseline_hpwl: Optional[float] = None,
) -> Dict[str, float]:
    """Compute all placement metrics and return as a dict."""
    hpwl = compute_hpwl(cell_x, cell_y, net_cell_indices, net_weights)
    mean_ovfl, top5_ovfl = compute_overflow(cell_x, cell_y, cell_w, cell_h, die_w, die_h)
    tns = compute_tns_proxy(cell_x, cell_y, net_cell_indices, critical_net_flags)

    metrics = {
        "hpwl": hpwl,
        "mean_overflow": mean_ovfl,
        "top5_overflow": top5_ovfl,
        "tns_proxy": tns,
    }

    if baseline_hpwl is not None:
        metrics["normalized_hpwl"] = normalize_hpwl(hpwl, baseline_hpwl)

    return metrics
