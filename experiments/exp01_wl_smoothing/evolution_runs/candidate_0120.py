import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-annealed gamma schedule for WA-WL placement."""

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:               # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base schedule: log-space cosine anneal high -> low --------------
    # Cosine easing keeps gamma high a bit longer (cells stay clustered),
    # then descends smoothly into the fine-tuning regime.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ----------------------------------------------
    # Overflow is the true physical signal of how spread-out the cells are.
    # When overflow is high we want smoother gradients (higher gamma); as the
    # layout legalizes (overflow -> 0) we trust the sharper HPWL approximation.
    # Blend a multiplicative term (scales the annealed base) with an additive
    # floor driven directly by overflow so gamma can't collapse while cells
    # are still badly overlapping early on.
    ov_mult = 0.60 + 1.40 * (ov ** 1.20)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.40)
    gamma = 0.60 * base * ov_mult + 0.40 * ov_add

    # --- HPWL feedback: react to convergence dynamics -------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            # Plateau: improvement has stalled -> sharpen to chase HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: HPWL climbing -> smooth gradients to recover.
            if last > first * 1.02:
                gamma *= 1.40
            # Healthy descent: nudge sharper to refine.
            elif last < first * 0.98:
                gamma *= 0.92

    # --- late-stage ceilings: force fine-tuning regime ------------------
    # Only clamp hard once the layout is reasonably legal (low overflow);
    # if overflow is still high we keep more smoothing headroom.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final guards ----------------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))