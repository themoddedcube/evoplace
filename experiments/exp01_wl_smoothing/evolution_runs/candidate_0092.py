import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL smoothing."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing: log-linear (geometric) decay shaped by cosine ---
    # cos_prog moves 0 -> 1 slowly at the start (keep cells clustered),
    # then accelerates toward the end (sharpen HPWL accuracy).
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # When density is still high we want smoother gradients (raise gamma);
    # as the layout legalizes (overflow -> 0) we trust the geometric decay.
    # Blend a multiplicative term (tracks the decay curve) with an additive
    # floor anchored directly to overflow so very high overflow never drops
    # gamma too low regardless of progress.
    ov_mult = 0.6 + 1.4 * (ov ** 1.2)
    ov_anchor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    gamma = 0.6 * base * ov_mult + 0.4 * ov_anchor

    # --- HPWL feedback: react to convergence dynamics ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Plateau: best HPWL barely improving -> sharpen (lower gamma)
            # to refine wirelength on the now-settled layout.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging / oscillating up -> smooth more (raise gamma)
            # to recover stable gradients.
            if last > first * 1.02:
                gamma *= 1.40
            # Healthy steady descent -> gently sharpen.
            elif last < first * 0.97:
                gamma *= 0.93

    # --- late-stage ceilings: force accuracy near the end ---
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final NaN guard + clamp ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))