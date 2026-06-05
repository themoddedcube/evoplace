"""
Path-group-aware timing loss for DREAMPlace global placement (Exp 4).

Provides two integration modes:

  Variant A (default, v1): writes static criticality weights to
  placedb.net_weights before NonLinearPlace runs. Zero extra GPU
  code — the existing WA-WL CUDA kernel multiplies by net_weights
  automatically.

  Variant B (hook-based, v2): returns a timing_loss callable matching
  dreamplace_ext/hooks.py's timing_loss hook signature. Used once the
  optimizer loop is patched to call our hooks each iteration.

Usage:
    from models.path_group_classifier import classify_nets
    from models.path_group_loss import apply_weights_variant_a, make_timing_hook

    pg_data = classify_nets(placedb, cell_type_map, sdc_path)
    apply_weights_variant_a(placedb, pg_data)          # Variant A
    # or
    hook_fn = make_timing_hook(pg_data)                # Variant B
    hooks.set_timing_loss(hook_fn)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from models.path_group_classifier import PathGroupData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant A — write weights to placedb.net_weights (v1, works now)
# ---------------------------------------------------------------------------

def apply_weights_variant_a(placedb, pg_data: PathGroupData) -> bool:
    """
    Write path-group criticality weights into placedb.net_weights.

    DREAMPlace's WA-WL kernel multiplies each net's half-perimeter by
    placedb.net_weights[net_id], so critical nets get proportionally
    higher gradient pressure toward co-location.

    Returns True if weights were applied, False if not applicable
    (e.g. no timing constraints, or placedb lacks net_weights attribute).
    """
    if not pg_data.has_timing_constraints:
        return False

    if not hasattr(placedb, "net_weights"):
        logger.warning("placedb has no net_weights attribute — Variant A skipped")
        return False

    try:
        weights = pg_data.net_weights
        current = placedb.net_weights

        if len(weights) != len(current):
            logger.warning(
                f"net_weights length mismatch: pg_data={len(weights)}, "
                f"placedb={len(current)} — Variant A skipped"
            )
            return False

        # Multiply into existing weights (DREAMPlace may have pre-set timing weights
        # from its own net weighting pass; we combine rather than overwrite)
        placedb.net_weights[:] = current * weights

        critical = int((weights > 1.05).sum())
        logger.info(
            f"Variant A: applied path-group weights to {critical} critical nets "
            f"(max weight = {weights.max():.2f})"
        )
        return True

    except Exception as e:
        logger.warning(f"Variant A weight application failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Variant B — timing_loss hook (v2, for when optimizer loop is patched)
# ---------------------------------------------------------------------------

def make_timing_hook(
    pg_data: PathGroupData,
    alpha: Optional[float] = None,
) -> callable:
    """
    Return a timing_loss callable for dreamplace_ext/hooks.set_timing_loss().

    Signature: fn(cell_x, cell_y, net_cell_indices, iteration)
               -> (loss_scalar, grad_x, grad_y)

    If pg_data.has_timing_constraints is False, returns a no-op that adds
    zero loss — placement is identical to unmodified DREAMPlace.

    alpha overrides pg_data.config.timing_loss_alpha if provided.
    """
    if alpha is None:
        alpha = pg_data.config.timing_loss_alpha

    if not pg_data.has_timing_constraints:
        def noop(cell_x, cell_y, net_cell_indices, iteration):
            zeros_x = np.zeros_like(np.asarray(cell_x))
            zeros_y = np.zeros_like(np.asarray(cell_y))
            return 0.0, zeros_x, zeros_y
        return noop

    cfg = pg_data.config
    weights_np = pg_data.net_weights.copy()

    def timing_loss_fn(
        cell_x,
        cell_y,
        net_cell_indices: List[List[int]],
        iteration: int,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        try:
            import torch
        except ImportError:
            zeros_x = np.zeros_like(np.asarray(cell_x))
            zeros_y = np.zeros_like(np.asarray(cell_y))
            return 0.0, zeros_x, zeros_y

        if not isinstance(cell_x, torch.Tensor):
            cx = torch.from_numpy(np.asarray(cell_x, dtype=np.float32)).requires_grad_(True)
            cy = torch.from_numpy(np.asarray(cell_y, dtype=np.float32)).requires_grad_(True)
        else:
            cx = cell_x.float().detach().requires_grad_(True)
            cy = cell_y.float().detach().requires_grad_(True)

        weights_t = torch.from_numpy(weights_np)
        # Only iterate over critical nets (weight > 1.05) for efficiency
        critical_ids = np.where(weights_np > 1.05)[0]

        total_loss = torch.zeros(1, dtype=torch.float32)
        gamma = cfg.gamma_timing

        for i in critical_ids:
            if i >= len(net_cell_indices):
                continue
            cells = net_cell_indices[i]
            if len(cells) < 2:
                continue
            w = float(weights_t[i])
            idx = torch.tensor(cells, dtype=torch.long)
            xs = cx[idx]
            ys = cy[idx]
            total_loss = total_loss + w * (_wa_wl_1d(xs, gamma) + _wa_wl_1d(ys, gamma))

        total_loss = alpha * total_loss
        total_loss.backward()

        gx = cx.grad.numpy() if cx.grad is not None else np.zeros(len(cell_x))
        gy = cy.grad.numpy() if cy.grad is not None else np.zeros(len(cell_y))
        return float(total_loss.item()), gx, gy

    return timing_loss_fn


def _wa_wl_1d(coords, gamma: float):
    """Differentiable weighted-average wirelength along one axis."""
    import torch
    exp_p = torch.exp(coords / gamma)
    exp_n = torch.exp(-coords / gamma)
    wl_p = (coords * exp_p).sum() / exp_p.sum()
    wl_n = (coords * exp_n).sum() / exp_n.sum()
    return wl_p - wl_n
