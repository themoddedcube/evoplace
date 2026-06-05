import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # Guard inputs
    T = max(1, int(total_iterations))
    t = min(max(0, int(iteration)), T)
    progress = t / T
    of = overflow if (overflow is not None and overflow == overflow) else 1.0
    of = min(1.0, max(0.0, of))

    # --- Base annealing: high gamma early, low gamma late ---
    # Cosine annealing between gamma_hi and gamma_lo on iteration progress.
    gamma_hi = 8.0
    gamma_lo = 0.5
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    base = gamma_lo + (gamma_hi - gamma_lo) * cos_factor

    # --- Overflow-adaptive multiplier ---
    # When cells are still spread (high overflow) keep gradients smooth (boost
    # gamma); once density settles (low overflow) sharpen toward accurate HPWL.
    # Multiplier ranges ~[0.6, 2.0] centered near the typical 0.1 target.
    of_mult = math.exp(1.0 * (of - 0.1))
    of_mult = min(2.0, max(0.6, of_mult))

    gamma = base * of_mult

    # --- HPWL plateau detection: if recent HPWL stops improving, sharpen ---
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-4:] if h is not None and h == h and h > 0]
        if len(recent) >= 4:
            prev = sum(recent[:2]) / 2.0
            curr = sum(recent[2:]) / 2.0
            if prev > 0 and (prev - curr) / prev < 1e-4:
                gamma *= 0.7  # plateau -> reduce gamma for finer HPWL fitting

    # Late-stage floor pull: ensure accurate HPWL in final fine-tuning.
    if progress > 0.85:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, float(gamma)))