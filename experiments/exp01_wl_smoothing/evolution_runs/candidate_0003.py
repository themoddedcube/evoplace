import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # --- sanitize inputs ---
    if total_iterations <= 0:
        total_iterations = 1
    ov = overflow
    if ov != ov:          # NaN guard
        ov = 1.0
    ov = min(1.0, max(0.0, ov))
    progress = min(1.0, max(0.0, float(iteration) / float(total_iterations)))

    # --- overflow-adaptive core (DREAMPlace-style) ---
    # Dominant signal: bins full -> high gamma (smooth), cells spread -> low gamma.
    gamma_ov = 4.0 * 10.0 ** ((ov - 0.1) * 20.0 / 9.0 - 1.0)

    # --- iteration cosine annealing 8.0 -> 0.5 ---
    # Guarantees continued smoothing even if overflow stalls.
    gamma_hi, gamma_lo = 8.0, 0.5
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    gamma_iter = gamma_lo + (gamma_hi - gamma_lo) * cos

    # --- blend: trust overflow early, lean on annealing for late fine-tuning ---
    w = progress
    gamma = (1.0 - w) * gamma_ov + w * min(gamma_ov, gamma_iter)

    # --- plateau detection: sharpen WL approximation when HPWL stops improving ---
    if len(hpwl_history) >= 6:
        recent = hpwl_history[-3:]
        prev = hpwl_history[-6:-3]
        if all(x == x for x in recent) and all(x == x for x in prev):
            r = sum(recent) / 3.0
            p = sum(prev) / 3.0
            if p > 0.0 and (p - r) / p < 1e-4:
                gamma *= 0.7      # accelerate descent toward accurate HPWL

    # --- late hard floor so final iterations optimize true wirelength ---
    if progress > 0.9:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))