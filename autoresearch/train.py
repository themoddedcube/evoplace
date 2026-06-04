"""
EDITABLE: Current best placement config under autoresearch.

This file is the ONLY file modified by the AI agent in the autoresearch loop.
The evaluate.py harness loads this file and calls the functions defined here.

RULES (AI agent must follow these):
- Define gamma_schedule and/or lambda_schedule with the exact signatures below
- Do NOT import external libraries (math, numpy are allowed)
- Do NOT add file I/O or network calls
- Return values must be in the specified ranges
- Leaving a function undefined = use DreamPlace's default

Current status: baseline (DreamPlace default schedules)
Last modified: 2026-06-04 (project init)
Best val_norm_hpwl seen: 1.0000 (baseline)
"""

import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """
    WA-WL smoothness (gamma) schedule.
    Returns gamma in [0.01, 20.0].
    Lower gamma = more accurate HPWL, noisier gradients.
    Higher gamma = smoother gradients, less accurate.
    """
    # Baseline: linear decay 8.0 → 0.5
    gamma_max = 8.0
    gamma_min = 0.5
    t = iteration / max(total_iterations - 1, 1)
    return gamma_max - (gamma_max - gamma_min) * t


def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """
    Density weight (lambda) schedule.
    Returns new lambda > 0 (typically grows over time).
    """
    # Baseline: exponential increase based on overflow level
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
