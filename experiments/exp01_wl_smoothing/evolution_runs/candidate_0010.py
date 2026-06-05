import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    t = iteration / max(1, total_iterations)
    of = min(1.0, max(0.0, overflow))

    # Overflow-adaptive base (DREAMPlace-style): high gamma while cells are
    # still spread across bins, collapsing as density resolves.
    overflow_gamma = 4.0 * 10.0 ** ((of - 0.1) * 20.0 / 9.0 - 1.0)

    # Progress-based annealing: cosine decay from smooth -> accurate so the
    # tail of optimization sees low gamma even if overflow plateaus.
    gamma_hi, gamma_lo = 8.0, 0.5
    cosine = gamma_lo + 0.5 * (gamma_hi - gamma_lo) * (1.0 + math.cos(math.pi * t))

    # Geometric blend keeps both signals influential without one dominating;
    # overflow drives the early clustering, the cosine guarantees fine-tuning.
    gamma = math.sqrt(overflow_gamma * cosine)

    # Plateau detection: if HPWL has stopped improving, sharpen the
    # approximation to push past the stall.
    if len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        if recent[-1] >= min(recent[:-1]) * 0.9999:
            gamma *= 0.7

    return min(50.0, max(0.01, gamma))