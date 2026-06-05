import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma with cosine progress annealing and HPWL-trajectory adaptation."""

    # --- robust input sanitation ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary signal: overflow-driven geometric interpolation (DREAMPlace-style) ---
    # overflow ~1.0 early (cells overlapped) -> smooth, high gamma
    # overflow ~0.0 late  (cells legalized)  -> accurate, low gamma
    # exponent < 1 holds gamma high while overflow is still moderate, aiding clustering.
    ov_exp = ov ** 0.65
    base = gamma_low * (gamma_high / gamma_low) ** ov_exp

    # --- progress envelope: guarantees annealing even if overflow lingers ---
    cos_env = 0.5 + 0.5 * math.cos(math.pi * progress)          # 1 -> 0
    prog_gamma = gamma_low + (gamma_high - gamma_low) * cos_env

    # overflow dominates the schedule; progress supplies a monotone anneal coupling.
    gamma = 0.70 * base + 0.30 * prog_gamma

    # --- plateau / divergence adaptation from HPWL trajectory ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # improvement stalled -> sharpen toward accurate HPWL
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # HPWL climbing (instability) -> smooth gradients to recover
            if last > first * 1.02:
                gamma *= 1.30
            # HPWL still descending nicely -> keep sharpening gently
            elif last < first * 0.98:
                gamma *= 0.93

    # --- late-stage accuracy ceilings (apply only once cells are spread) ---
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final guards ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))