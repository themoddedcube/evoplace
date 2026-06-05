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
    # Overflow-driven base, matching DREAMPlace's coarse-to-fine intent:
    # strong smoothing while spreading, sharp WL accuracy near convergence.
    base = 4.0 * 10.0 ** ((overflow - 0.1) * 20.0 / 9.0 - 1.0)

    # Iteration progress in [0, 1].
    progress = 0.0
    if total_iterations > 1:
        progress = min(1.0, max(0.0, iteration / (total_iterations - 1)))

    # Late-stage sharpening. Overflow alone can plateau above 0.1 while the
    # placement is effectively legalized; couple a convergence signal with the
    # iteration clock so gamma keeps annealing toward a sharper WL estimate.
    # conv ramps up as overflow drops below the 0.1 knee point.
    conv = max(0.0, min(1.0, (0.15 - overflow) / 0.15))
    sharpen = 0.6 * conv + 0.4 * (progress ** 1.5)
    gamma = base * (1.0 - 0.40 * sharpen)

    # Plateau detection: if HPWL has effectively stopped improving over the
    # recent window, crisp up the gradients (lower gamma) to let the optimizer
    # resolve fine wirelength structure instead of sitting on a smoothed basin.
    if len(hpwl_history) >= 5:
        recent = hpwl_history[-1]
        ref = hpwl_history[-5]
        if ref > 0.0:
            rel_gain = (ref - recent) / ref
            if rel_gain < 1.0e-4:
                gamma *= 0.80
            elif rel_gain < 5.0e-4:
                gamma *= 0.92

    # Guard a small floor early on so spreading gradients stay smooth even if
    # the signals above push gamma down prematurely.
    if overflow > 0.5:
        gamma = max(gamma, 8.0)

    return min(50.0, max(0.01, gamma))