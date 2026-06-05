import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """ ... """

    # Normalized progress in [0, 1]
    t = 0.0
    if total_iterations > 1:
        t = iteration / float(total_iterations - 1)
    t = min(1.0, max(0.0, t))

    # Clamp overflow to a sane range
    ov = min(1.0, max(0.0, overflow))

    # --- Base schedule: high gamma early, low gamma late ---
    # Cosine annealing between gamma_hi and gamma_lo over progress.
    gamma_hi = 8.0
    gamma_lo = 0.5
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
    base = gamma_lo + (gamma_hi - gamma_lo) * cos_factor

    # --- Overflow-adaptive coupling (DREAMPlace-style) ---
    # While cells are still spread out (high overflow) keep gradients smooth;
    # as the layout legalizes (overflow drops) sharpen toward accurate HPWL.
    # Exponential map keeps gamma high until overflow falls below ~0.1.
    ov_factor = 10.0 ** ((ov - 0.1) * 20.0 / 9.0 - 1.0)  # ~ in [0.1, ~6.5]
    overflow_gamma = 4.0 * ov_factor

    # Blend: trust overflow more once we are far enough into the run,
    # but never let gamma spike back up late even if overflow stalls.
    w = t  # weight on the progress-driven (decaying) component
    gamma = (1.0 - w) * overflow_gamma + w * base

    # Take the smaller of the blended value and a hard progress cap so that
    # late iterations are guaranteed to fine-tune at low gamma.
    late_cap = gamma_lo + (gamma_hi - gamma_lo) * cos_factor
    gamma = min(gamma, max(late_cap, overflow_gamma * (1.0 - 0.5 * t)))

    # --- Plateau detection: if HPWL has stopped improving, sharpen faster ---
    if len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        prev = recent[0]
        improving = False
        for h in recent[1:]:
            if h < prev * (1.0 - 1e-4):
                improving = True
            prev = h
        if not improving:
            gamma *= 0.7  # push toward accurate-HPWL regime

    return min(50.0, max(0.01, gamma))