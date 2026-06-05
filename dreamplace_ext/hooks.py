"""
DreamPlace hook injection layer.

Maintains singleton callables that patch into DreamPlace's internals
during a placement run. All hooks are reset between runs.

DreamPlace integration points:
- gamma_schedule: called each iteration to get the WA-WL smoothness γ
- lambda_schedule: called each iteration to get the density weight λ
- init_positions: called once before global placement to set starting positions
- timing_loss: called each iteration to add a timing term to the loss

When a hook is None, DreamPlace uses its default heuristic behavior.
"""

from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Hook registry (module-level singletons)
# ---------------------------------------------------------------------------

_gamma_schedule_fn: Optional[Callable] = None
_lambda_schedule_fn: Optional[Callable] = None
_init_positions_fn: Optional[Callable] = None
_timing_loss_fn: Optional[Callable] = None
_path_group_data = None   # PathGroupData, set once per run before NonLinearPlace
_divergence_count: int = 0

# Deferred net-weight activation (path-group timing weights, Exp 4).
# Weights are stored here and applied to data_collections.net_weights only once
# overflow drops below _deferred_activation_overflow (default 0.3).  This avoids
# biasing early global spreading toward timing-critical nets.
_deferred_net_weights = None   # numpy float32 array, same length as net_weights
_deferred_activation_overflow: float = 0.3
_deferred_weights_applied: bool = False


def set_gamma_schedule(fn: Optional[Callable]):
    """
    Set the WA-WL smoothness schedule.

    Signature: fn(iteration: int, total_iterations: int,
                  overflow: float, hpwl_history: list) -> float
    Returns γ ∈ (0, ∞), dimensionless (scaled by bin size inside PlaceObj).
    DreamPlace default: overflow-driven, 4.0 * 10^((overflow-0.1)*20/9 - 1).
    """
    global _gamma_schedule_fn
    _gamma_schedule_fn = fn


def set_lambda_schedule(fn: Optional[Callable]):
    """
    Set the density weight schedule.

    Signature: fn(iteration: int, overflow: float,
                  overflow_history: list, gradient_norm: float,
                  current_lambda: float) -> float
    Returns λ > 0. DreamPlace default: exponential increase based on density.
    """
    global _lambda_schedule_fn
    _lambda_schedule_fn = fn


def set_init_positions(fn: Optional[Callable]):
    """
    Set the initialization function.

    Signature: fn(place_db) -> (cell_x: np.ndarray, cell_y: np.ndarray)
    cell_x, cell_y normalized to [0, 1]. DreamPlace default: uniform center.
    """
    global _init_positions_fn
    _init_positions_fn = fn


def set_timing_loss(fn: Optional[Callable]):
    """
    Set the timing surrogate loss term.

    Signature: fn(cell_x, cell_y, net_cell_indices, iteration) -> (loss_scalar, grad_x, grad_y)
    All tensors should be PyTorch tensors on GPU. Return 0-loss if not applicable.
    """
    global _timing_loss_fn
    _timing_loss_fn = fn


def get_gamma_schedule() -> Optional[Callable]:
    return _gamma_schedule_fn


def get_lambda_schedule() -> Optional[Callable]:
    return _lambda_schedule_fn


def get_init_positions() -> Optional[Callable]:
    return _init_positions_fn


def get_timing_loss() -> Optional[Callable]:
    return _timing_loss_fn


def set_path_group_data(pg_data):
    """Store PathGroupData for the current run (for inspection/logging)."""
    global _path_group_data
    _path_group_data = pg_data


def get_path_group_data():
    return _path_group_data


def set_deferred_net_weights(weights_np, threshold: float = 0.3):
    """
    Register path-group net weights for phased activation.

    Weights are written to data_collections.net_weights the first time
    overflow drops below `threshold` (called from PlaceObj.update_gamma).
    Pass threshold=0.0 to apply immediately on first iteration.
    """
    global _deferred_net_weights, _deferred_activation_overflow, _deferred_weights_applied
    _deferred_net_weights = weights_np
    _deferred_activation_overflow = threshold
    _deferred_weights_applied = False


def pop_deferred_net_weights_if_triggered(overflow: float):
    """
    Return the deferred net weights if overflow crossed the threshold, else None.

    Returns the weight array exactly once (marks applied on first trigger so
    subsequent calls return None). Called each iteration from PlaceObj.update_gamma.
    """
    global _deferred_weights_applied, _deferred_net_weights
    if _deferred_net_weights is None or _deferred_weights_applied:
        return None
    if overflow <= _deferred_activation_overflow:
        _deferred_weights_applied = True
        return _deferred_net_weights
    return None


def record_divergence():
    global _divergence_count
    _divergence_count += 1


def get_divergence_count() -> int:
    return _divergence_count


def reset():
    """Reset all hooks and counters. Call between placement runs."""
    global _gamma_schedule_fn, _lambda_schedule_fn, _init_positions_fn
    global _timing_loss_fn, _path_group_data, _divergence_count
    global _deferred_net_weights, _deferred_activation_overflow, _deferred_weights_applied
    _gamma_schedule_fn = None
    _lambda_schedule_fn = None
    _init_positions_fn = None
    _timing_loss_fn = None
    _path_group_data = None
    _divergence_count = 0
    _deferred_net_weights = None
    _deferred_activation_overflow = 0.3
    _deferred_weights_applied = False
