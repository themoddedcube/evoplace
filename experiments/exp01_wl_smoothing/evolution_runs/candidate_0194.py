import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware log-cosine gamma schedule with plateau adaptation."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # --- Base decay: cosine annealing in log-space (smooth high->low) ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = math.exp(log_hi + (log_lo - log_hi) * cos_prog)

    # --- Overflow coupling -------------------------------------------------
    # Physical truth: while cells still overlap (high overflow), keeping gamma
    # higher preserves smooth, well-conditioned gradients. As the layout
    # legalizes (overflow -> 0) we trust the sharp HPWL approximation.
    # Blend a multiplicative term (scales the decay) with an additive floor
    # (guarantees enough smoothing when massively overflowed).
    ov_s = ov ** 1.3
    ov_mult = 0.50 + 1.50 * ov_s
    ov_floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.6)
    gamma = 0.60 * base * ov_mult + 0.40 * ov_floor

    # --- HPWL-history feedback --------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Stagnation: best-of-window barely improved vs the older point.
            # Sharpen gamma to chase a more accurate wirelength minimum.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: HPWL climbing -> gradients too sharp/noisy, smooth back.
            if last > first * 1.02:
                gamma *= 1.40
            # Healthy descent: nudge sharper to refine.
            elif last < first * 0.97:
                gamma *= 0.93

    # --- Late-stage ceilings: force accurate HPWL once layout is settling --
    if progress > 0.90:
        ceil = 1.2 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.80:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.1)
    elif progress > 0.65:
        gamma = min(gamma, 3.0 if ov > 0.12 else 1.8)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))