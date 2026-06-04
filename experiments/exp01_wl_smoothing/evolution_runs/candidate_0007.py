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

    # Reference overflow at which gamma_min is reached
    ov_ref = 0.05
    ov = max(ov_ref, min(1.0, overflow))

    # Log-scale overflow factor: maps [ov_ref, 1.0] -> [0, 1]
    # More sensitive to changes in the low-overflow fine-tuning regime
    log_factor = 1.0 - math.log(ov) / math.log(ov_ref)
    log_factor = max(0.0, min(1.0, log_factor))
    overflow_gamma = gamma_min + (gamma_max - gamma_min) * log_factor

    # Cosine annealing: smooth monotonic time-based decay
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t))
    time_gamma = gamma_min + (gamma_max - gamma_min) * cosine_factor

    # Adaptive blend: overflow-heavy early (cells still spreading),
    # time-heavy late (enforce convergence regardless of overflow noise)
    w_overflow = max(0.25, 1.0 - t)
    gamma = w_overflow * overflow_gamma + (1.0 - w_overflow) * time_gamma

    return max(0.01, min(20.0, gamma))