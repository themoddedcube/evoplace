"""
Baseline and evolved scheduler implementations.

Each scheduler must match the hook signature defined in hooks.py.
These are starting points for OpenEvolve to mutate.
"""

import math
from typing import List


# ---------------------------------------------------------------------------
# Gamma (WA-WL smoothness) schedules
# ---------------------------------------------------------------------------

def gamma_baseline_linear(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: List[float],
) -> float:
    """
    DREAMPlace default: linear decay from 8.0 to 0.5.
    This is the baseline that evolution should beat.
    """
    gamma_max = 8.0
    gamma_min = 0.5
    t = iteration / max(total_iterations, 1)
    return gamma_max - (gamma_max - gamma_min) * t


def gamma_exponential(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: List[float],
) -> float:
    """Exponential decay: faster decrease early, slower at end."""
    gamma_max = 8.0
    gamma_min = 0.5
    t = iteration / max(total_iterations, 1)
    decay = math.exp(-5 * t)
    return gamma_min + (gamma_max - gamma_min) * decay


def gamma_overflow_adaptive(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: List[float],
) -> float:
    """
    Overflow-adaptive: decrease gamma faster when overflow is high
    (cells need more accurate wirelength signal to pull them together).
    """
    gamma_max = 8.0
    gamma_min = 0.5
    # Scale inversely with overflow: high overflow → low gamma → accurate WL gradients
    overflow_clamped = max(0.05, min(1.0, overflow))
    t = iteration / max(total_iterations, 1)
    base = gamma_min + (gamma_max - gamma_min) * (1 - t)
    return base * overflow_clamped


def gamma_cosine_annealing(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: List[float],
) -> float:
    """Cosine annealing: smooth monotone decrease via cosine schedule."""
    gamma_max = 8.0
    gamma_min = 0.5
    t = iteration / max(total_iterations, 1)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * t))
    return gamma_min + (gamma_max - gamma_min) * cosine_factor


# ---------------------------------------------------------------------------
# Lambda (density weight) schedules
# ---------------------------------------------------------------------------

def lambda_baseline_exponential(
    iteration: int,
    overflow: float,
    overflow_history: List[float],
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """
    DREAMPlace default: exponential increase based on overflow level.
    This is the baseline that evolution should beat.
    """
    lambda_max = 1e6
    if overflow > 0.9:
        multiplier = 1.05
    elif overflow > 0.5:
        multiplier = 1.02
    elif overflow > 0.2:
        multiplier = 1.01
    else:
        multiplier = 1.005
    return min(lambda_max, current_lambda * multiplier)


def lambda_gradient_adaptive(
    iteration: int,
    overflow: float,
    overflow_history: List[float],
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """
    Gradient-norm adaptive: increase lambda more aggressively when gradient
    norm is small (optimizer is in flat region / near saddle point).
    """
    lambda_max = 1e6
    base_multiplier = 1.02
    if gradient_norm < 1e-3:
        # Flat gradient = stuck; push harder
        multiplier = 1.1
    elif gradient_norm > 1.0:
        # Large gradient = good progress; don't push too hard
        multiplier = 1.005
    else:
        multiplier = base_multiplier
    # Also factor in overflow
    overflow_factor = max(0.5, min(2.0, overflow * 2))
    multiplier = 1.0 + (multiplier - 1.0) * overflow_factor
    return min(lambda_max, current_lambda * multiplier)


def lambda_plateau_detector(
    iteration: int,
    overflow: float,
    overflow_history: List[float],
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """
    Plateau-aware: detect overflow plateau and apply quadratic boost.
    Mirrors DREAMPlace 3.0's entropy injection but at the λ level.
    """
    lambda_max = 1e6
    window = 20
    if len(overflow_history) >= window and overflow > 0.9:
        window_data = overflow_history[-window:]
        range_ovfl = max(window_data) - min(window_data)
        avg_ovfl = sum(window_data) / len(window_data)
        # Plateau condition from DREAMPlace 3.0 paper
        if avg_ovfl > 0 and (range_ovfl / avg_ovfl) < 0.02:
            # Plateau detected: double lambda
            return min(lambda_max, current_lambda * 2.0)
    return min(lambda_max, current_lambda * 1.02)
