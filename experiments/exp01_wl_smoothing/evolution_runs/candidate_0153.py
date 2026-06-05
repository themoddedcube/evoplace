import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Base anneal: geometric decay along a cosine-shaped progress curve.
    # Stays high while cells are still spreading, drops smoothly for fine-tuning.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow coupling: physical placement state matters more than the raw
    # iteration count. While density is high, keep smoothing strong so cells can
    # keep moving; once bins clear, let gamma fall to sharpen the HPWL estimate.
    # Blend a multiplicative term (scales the anneal) with an additive floor
    # (guarantees enough smoothing when overflow is genuinely high).
    ov_mult = 0.6 + 1.4 * (ov ** 1.2)
    ov_floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    gamma = 0.6 * base * ov_mult + 0.4 * ov_floor

    # --- HPWL-history feedback: react to the actual optimization trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Plateau: improvement has stalled -> sharpen to chase real HPWL,
            # but only once density has mostly resolved (avoid freezing spread).
            if prev > 0 and (prev - best_recent) / prev < 1e-3 and ov < 0.20:
                gamma *= 0.80

            # Divergence: HPWL climbing -> re-smooth to stabilize gradients.
            if first > 0 and last > first * 1.02:
                gamma *= 1.30
            # Healthy descent -> gently sharpen to lock in gains.
            elif first > 0 and last < first * 0.985:
                gamma *= 0.93

    # --- Late-stage caps: in the final phase HPWL accuracy dominates, so force
    # gamma low unless overflow says cells still need room to move.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))