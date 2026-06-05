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
    # Clamp overflow defensively; the schedule is driven mainly by it.
    ovf = min(1.0, max(0.0, overflow))

    # Overflow-driven base (DREAMPlace family): smooth gradients while cells
    # are still spread out, sharpening as the layout legalizes.
    base = 4.0 * 10.0 ** ((ovf - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Late-stage sharpening: once the placement is mostly legal, bias gamma
    # downward so the WA model tracks true HPWL more closely. Use a smooth
    # (squared) legality gate so the transition out of the spreading phase is
    # gradual and never destabilizes the still-overflowing mid phase.
    legal = min(1.0, max(0.0, 1.0 - ovf / 0.18))
    legal_smooth = legal * legal
    sharpen = 1.0 - 0.55 * legal_smooth * (prog ** 1.3)

    # Endgame extra sharpen: in the final fifth of the run, when the layout is
    # nearly legal, push gamma further toward true HPWL. This is the regime
    # where the WA approximation error directly inflates measured HPWL.
    if prog > 0.8 and ovf < 0.10:
        tail = (prog - 0.8) / 0.2          # 0 -> 1 over the last 20%
        near = 1.0 - ovf / 0.10            # 1 at zero overflow
        sharpen *= 1.0 - 0.25 * tail * near

    gamma = base * sharpen

    # History-driven adaptation: react to the recent HPWL trajectory.
    if len(hpwl_history) >= 4:
        h = hpwl_history[-4:]
        recent = h[-1]
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev
            if -0.0015 < rel < 0.0015:
                # Plateau: tighten the WL approximation to squeeze out HPWL,
                # but only once the layout is legal enough to tolerate sharp
                # (noisier) gradients.
                gamma *= 0.80 if ovf < 0.15 else 0.92
            elif rel < -0.005:
                # Worsening / oscillating: restore smoothness to recover.
                gamma *= 1.18

    # Keep a small floor on gamma to avoid runaway gradient noise even when
    # fully sharpened in the endgame.
    floor = 0.05 if prog > 0.9 else 0.10
    return min(50.0, max(floor, gamma))