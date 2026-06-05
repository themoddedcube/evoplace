import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # --- sanitize inputs (avoid NaN/inf propagating into gamma) ---
    T = total_iterations if total_iterations and total_iterations > 0 else 1
    t = iteration if iteration and iteration > 0 else 0
    progress = t / float(T)
    if progress < 0.0:
        progress = 0.0
    elif progress > 1.0:
        progress = 1.0

    ovf = overflow
    if ovf != ovf or ovf < 0.0:      # NaN guard / clamp
        ovf = 0.0
    elif ovf > 1.0:
        ovf = 1.0

    # --- primary driver: cosine anneal from smooth -> accurate ---
    # high gamma early (cells still clustering), low gamma late (fine-tune HPWL).
    gamma_hi = 8.0
    gamma_lo = 0.5
    cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    gamma = gamma_lo + (gamma_hi - gamma_lo) * cos_term

    # --- overflow modulation: more spreading still needed => keep smoother ---
    # scaled so the low-overflow regime of this benchmark still gets a gentle push.
    gamma *= (1.0 + 2.0 * ovf)

    # --- plateau detection: if HPWL stalls, sharpen (lower gamma) for accuracy ---
    if hpwl_history and len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        prev = recent[0]
        cur = recent[-1]
        if prev == prev and cur == cur and prev > 0.0:   # finite, positive
            rel_improve = (prev - cur) / prev
            if rel_improve < 1e-4:                        # essentially flat
                gamma *= 0.7

    # --- final safety clamp ---
    if gamma != gamma:        # NaN -> safe mid value
        gamma = 1.0
    return min(50.0, max(0.01, gamma))