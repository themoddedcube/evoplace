""" ... """

import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """ ... """

    t = iteration / max(1, total_iterations)
    ov = min(1.0, max(0.0, overflow))

    # Overflow-adaptive base (DREAMPlace-style): high gamma while bins are
    # over-dense (cells still clustering), low gamma once density relaxes.
    gamma_ov = 4.0 * 10.0 ** ((ov - 0.1) * 20.0 / 9.0 - 1.0)

    # Iteration-driven cosine annealing from high -> low. Acts as a floor so
    # gamma still decays for fine-tuning even if overflow stalls high.
    gamma_hi, gamma_lo = 8.0, 0.5
    cos = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0 over the run
    gamma_iter = gamma_lo + (gamma_hi - gamma_lo) * cos

    # Blend in log-space: trust overflow early, lean on the iteration floor late.
    w = t
    gamma = gamma_ov ** (1.0 - w) * gamma_iter ** w

    # Plateau detection: if HPWL has stagnated, sharpen the approximation
    # (lower gamma) to chase a more accurate, finer-grained optimum.
    if len(hpwl_history) >= 6:
        recent = sum(hpwl_history[-3:]) / 3.0
        prev = sum(hpwl_history[-6:-3]) / 3.0
        if prev > 0.0 and abs(prev - recent) / prev < 1e-3:
            gamma *= 0.7

    # Late-stage guard: force accurate HPWL approximation near convergence.
    if t > 0.85 and ov < 0.15:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))