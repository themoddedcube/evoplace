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


def set_gamma_schedule(fn: Optional[Callable]):
    """
    Set the WA-WL smoothness schedule.

    Signature: fn(iteration: int, total_iterations: int,
                  overflow: float, hpwl_history: list) -> float
    Returns γ ∈ (0, ∞). DreamPlace default: linear decay 8.0 → 0.5.
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


def record_divergence():
    global _divergence_count
    _divergence_count += 1


def get_divergence_count() -> int:
    return _divergence_count


def reset():
    """Reset all hooks and counters. Call between placement runs."""
    global _gamma_schedule_fn, _lambda_schedule_fn, _init_positions_fn
    global _timing_loss_fn, _path_group_data, _divergence_count
    _gamma_schedule_fn = None
    _lambda_schedule_fn = None
    _init_positions_fn = None
    _timing_loss_fn = None
    _path_group_data = None
    _divergence_count = 0
