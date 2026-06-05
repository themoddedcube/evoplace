import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware geometric gamma anneal with plateau/divergence control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Base anneal: geometric decay on a cosine-shaped progress curve.
    # Spends extra time at high gamma early (cells still clustering) and
    # eases into low gamma late for accurate HPWL.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow coupling. High overflow => keep gradients smooth (raise
    # gamma); low overflow (cells nearly legal) => let gamma drop for sharp
    # wirelength. Blend a multiplicative and an additive overflow term so the
    # schedule never collapses to ~0 while bins are still congested.
    ov_mult = 0.60 + 1.5 * (ov ** 1.20)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = 0.55 * base * ov_mult + 0.45 * ov_add

    # --- HPWL feedback: react to plateaus and divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Plateau: best is barely improving -> sharpen to escape.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: HPWL climbing -> smooth gradients to recover.
            if first > 0 and last > first * 1.02:
                gamma *= 1.40
            # Healthy descent: nudge sharper to lock in wirelength.
            elif first > 0 and last < first * 0.98:
                gamma *= 0.93

    # --- Late-stage ceilings: force fine-tuning regime, but stay smoother
    # while overflow is still non-trivial.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))