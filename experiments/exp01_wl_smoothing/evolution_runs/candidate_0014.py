""" ... """

import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """ ... """

    gamma_max = 8.0
    gamma_min = 0.5
    t = iteration / max(total_iterations - 1, 1)

    # Overflow-adaptive component: high overflow → high gamma (smooth gradients
    # for spreading), low overflow → low gamma (accurate HPWL for fine-tuning).
    # Power < 1 keeps gamma elevated through mid-overflow, drops sharply near 0.
    ov = max(0.0, min(1.0, overflow))
    ov_gamma = gamma_min + (gamma_max - gamma_min) * (ov ** 0.75)

    # Exponential time decay: ensures monotonic decrease even if overflow
    # measurement lags or plateaus — stays higher early, drops faster late.
    time_gamma = gamma_min + (gamma_max - gamma_min) * math.exp(-2.5 * t)

    # Geometric mean blends both signals: neither dominates completely.
    gamma = math.sqrt(ov_gamma * time_gamma)

    # HPWL convergence detection: if improvement per step has stalled, reduce
    # gamma to sharpen the WA-WL approximation and escape the plateau.
    if len(hpwl_history) >= 8:
        window = min(8, len(hpwl_history))
        h_now = hpwl_history[-1]
        h_past = hpwl_history[-window]
        if h_past > 0:
            per_iter_improvement = (h_past - h_now) / (h_past * window)
            if per_iter_improvement < 1e-4:
                gamma = max(gamma_min, gamma * 0.75)

    return max(gamma_min, min(20.0, gamma))