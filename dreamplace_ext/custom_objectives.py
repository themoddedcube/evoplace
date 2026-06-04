"""
Custom loss function components for DreamPlace.

These are the pluggable objective components that OpenEvolve will evolve.
Each must be self-contained (no external state) so they can be safely
serialized, mutated, and evaluated by the evolution engine.
"""

import math
from typing import List, Optional


# ---------------------------------------------------------------------------
# WA-WL smoothing functions
# ---------------------------------------------------------------------------

def wa_wirelength(
    x: "torch.Tensor",  # type: ignore
    gamma: float,
) -> "torch.Tensor":  # type: ignore
    """
    Weighted-average wirelength approximation.

    WL_WA(e; γ) = [Σ xᵢ·exp(xᵢ/γ) / Σ exp(xᵢ/γ)] - [Σ xᵢ·exp(-xᵢ/γ) / Σ exp(-xᵢ/γ)]

    As γ → 0: WL_WA → true HPWL (non-differentiable)
    As γ → ∞: WL_WA → center-of-mass (smooth but inaccurate)

    Args:
        x: (N,) tensor of cell positions along one axis
        gamma: smoothness parameter

    Returns:
        Scalar WA-WL value for this axis
    """
    import torch
    x_exp = torch.exp(x / gamma)
    x_neg_exp = torch.exp(-x / gamma)
    wl_pos = (x * x_exp).sum() / x_exp.sum()
    wl_neg = (x * x_neg_exp).sum() / x_neg_exp.sum()
    return wl_pos - wl_neg


# ---------------------------------------------------------------------------
# Timing surrogate (placeholder until Exp 4 trains the MLP)
# ---------------------------------------------------------------------------

def timing_loss_zero(cell_x, cell_y, net_cell_indices, iteration) -> tuple:
    """No-op timing loss. Returns (0.0, zero_grad_x, zero_grad_y)."""
    import numpy as np
    zero_x = np.zeros_like(cell_x)
    zero_y = np.zeros_like(cell_y)
    return 0.0, zero_x, zero_y


def timing_loss_hpwl_critical(cell_x, cell_y, net_cell_indices, iteration,
                               critical_net_flags=None, weight: float = 0.1) -> tuple:
    """
    Simple timing proxy: penalize HPWL of nets flagged as timing-critical.

    This is a differentiable approximation until the MLP surrogate is trained.
    """
    import numpy as np

    if critical_net_flags is None:
        critical_net_flags = [len(c) >= 4 for c in net_cell_indices]

    grad_x = np.zeros_like(cell_x)
    grad_y = np.zeros_like(cell_y)
    total_loss = 0.0

    for i, (cells, is_critical) in enumerate(zip(net_cell_indices, critical_net_flags)):
        if not is_critical or len(cells) < 2:
            continue
        xs = cell_x[cells]
        ys = cell_y[cells]
        hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
        total_loss += weight * hpwl

        # Subgradient: push max-x cell left, min-x cell right, etc.
        ix_max = cells[xs.argmax()]
        ix_min = cells[xs.argmin()]
        iy_max = cells[ys.argmax()]
        iy_min = cells[ys.argmin()]
        grad_x[ix_max] += weight
        grad_x[ix_min] -= weight
        grad_y[iy_max] += weight
        grad_y[iy_min] -= weight

    return total_loss, grad_x, grad_y
