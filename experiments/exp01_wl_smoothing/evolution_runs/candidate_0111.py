import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

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

    # --- base annealing: smooth log-space cosine decay high -> low ---
    # cos_prog goes 0 -> 1; geometric interpolation keeps gamma positive.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # While density overflow is high, cells are still spreading: keep gamma
    # elevated for smooth gradients. As overflow drains, trust the base
    # schedule and let gamma fall for an accurate HPWL approximation.
    # Blend weight shifts from overflow-driven (early) to schedule-driven (late).
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.3)
    w_ov = (1.0 - progress) * ov  # influence of overflow fades with progress
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # --- HPWL-feedback nudges (gentle, bounded) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Stalled improvement -> sharpen approximation to escape plateau.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging (HPWL climbing) -> smooth gradients to re-stabilize.
            if last > first * 1.02:
                gamma *= 1.30
            # Healthy descent -> ease gamma down a touch for refinement.
            elif last < first * 0.97:
                gamma *= 0.93

    # --- late-stage ceilings for fine-tuning (relaxed if still congested) ---
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final clamp ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))