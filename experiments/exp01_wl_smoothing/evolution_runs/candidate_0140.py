import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-annealed gamma schedule for WA-WL placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Base annealing: log-cosine from high -> low gamma -----------------
    # Cosine eases slowly at both ends: keeps gamma high while cells spread,
    # then plunges for accurate HPWL in the final fine-tuning regime.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow coupling -------------------------------------------------
    # Overflow is the true driver of how "spread" the layout is. When bins
    # are still congested we want smooth gradients (high gamma) regardless of
    # iteration count; once overflow collapses we trust the annealing floor.
    # Blend a multiplicative term (scales the schedule) with an additive
    # overflow target (anchors a sensible gamma from physics, not the clock).
    ov_mult = 0.6 + 1.4 * (ov ** 1.2)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    gamma = 0.5 * base * ov_mult + 0.5 * ov_add

    # --- HPWL feedback: react to the optimization trajectory ---------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stagnation: best HPWL barely improving -> sharpen for accuracy.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: HPWL climbing -> smooth gradients to recover.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.40
            # Steady descent: nudge sharper to lock in the gain.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.92

    # --- Late-stage ceilings: force accuracy near convergence --------------
    if progress > 0.90:
        ceil = 1.2 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.75:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.2)
    elif progress > 0.55:
        gamma = min(gamma, 4.0 if ov > 0.15 else 2.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))