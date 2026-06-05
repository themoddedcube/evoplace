"""
EVOLVE_TARGET: gamma_schedule

Initial seed program for OpenEvolve (Experiment 1: WL Smoothing Schedule).

This is the EDITABLE function that the LLM evolution engine will mutate.
The evaluation harness (evaluator_wrapper.py) calls this function and
measures the resulting HPWL on the benchmark.

RULES (do not change these comments — the evolution engine reads them):
- Function signature must be preserved exactly
- Only modify the function body
- No new imports allowed
- No external state or file I/O
- Return value must be a float in range [0.01, 50.0]
"""

import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """
    WA-WL smoothness schedule: returns γ for the weighted-average wirelength model.

    γ controls the tradeoff between WL accuracy and gradient smoothness:
    - High γ (~40): smooth gradients, inaccurate HPWL approximation
    - Low γ (~0.4): accurate HPWL, but gradients become noisy near convergence

    Args:
        iteration: current iteration number (0 to total_iterations-1)
        total_iterations: total planned iterations
        overflow: current density overflow (0.0 = no overflow, 1.0 = full overflow)
        hpwl_history: list of HPWL values at previous iterations

    Returns:
        gamma: float in [0.01, 50.0]
    """
    # Clamp overflow to the physically meaningful range up front.
    ov = min(1.0, max(0.0, overflow))

    # Overflow-driven base (DREAMPlace family): smooth gradients while cells
    # are still spread out, sharpening as the layout legalizes. Same calibrated
    # 0.4..40 envelope as the canonical WA schedule.
    base = 4.0 * 10.0 ** ((ov - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Late-stage sharpening: once the placement is mostly legal, bias gamma
    # downward so the WA model tracks true HPWL more closely. The push grows
    # with progress but is gated by low overflow so it never destabilizes the
    # still-spreading early/mid phases. Gating uses a smooth (cosine) ramp on
    # overflow rather than a hard cutoff, and the strength is a touch larger
    # because accurate WL near convergence is where HPWL is actually won.
    if ov < 0.18:
        legal = 0.5 * (1.0 + math.cos(math.pi * ov / 0.18))  # 1 at ov=0 -> 0 at ov=0.18
    else:
        legal = 0.0
    sharpen = 1.0 - 0.55 * legal * (prog ** 1.4)
    gamma = base * sharpen

    # History-driven adaptation: react to the recent HPWL trajectory using a
    # smoothed slope over a longer window so single-step noise does not flip
    # the response. The cumulative multiplier is clamped to keep gamma from
    # drifting too far off the overflow-scheduled value.
    if len(hpwl_history) >= 5:
        h = hpwl_history[-5:]
        recent = (h[-1] + h[-2]) / 2.0
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev  # >0 means improving
            if -0.0008 < rel < 0.0008:
                # Plateau: tighten the WL approximation to squeeze out HPWL,
                # but only meaningfully once the layout is reasonably legal.
                tighten = 0.80 + 0.12 * min(1.0, ov / 0.2)
                gamma *= tighten
            elif rel < -0.004:
                # Worsening / oscillating: restore smoothness to recover,
                # scaled by how badly it is regressing.
                boost = 1.0 + min(0.30, 0.15 + (-rel) * 8.0)
                gamma *= boost
            elif rel > 0.02:
                # Improving strongly: nudge gamma down to keep the accurate
                # WL pressure that is currently paying off.
                gamma *= 0.95

    return min(50.0, max(0.01, gamma))