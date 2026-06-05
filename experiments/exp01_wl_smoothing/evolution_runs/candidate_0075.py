import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base: cosine-annealed geometric decay over progress ---
    # cosine easing keeps gamma high a bit longer, then drops smoothly.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling: density is the real driver of needed smoothness ---
    # when cells are still spread (high overflow) we need smooth gradients;
    # once the layout has settled (low overflow) we can sharpen for accuracy.
    overflow_factor = 0.5 + 2.0 * (ov ** 1.2)

    # blend a progress-driven term with an overflow-driven term so neither
    # alone destabilizes the schedule.
    gamma = 0.5 * base * overflow_factor + 0.5 * (gamma_low + (gamma_high - gamma_low) * ov)

    # --- HPWL feedback: gently adapt to convergence behavior ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # stagnation: sharpen gamma to chase a more accurate optimum
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # divergence / oscillation: smooth gradients back out
            if window[-1] > window[0] * 1.02:
                gamma *= 1.3

    # --- late-stage accuracy clamp, but only once density is reasonable ---
    if progress > 0.80:
        ceil = 1.5 if ov > 0.10 else 0.8
        gamma = min(gamma, ceil)

    return min(50.0, max(0.01, gamma))