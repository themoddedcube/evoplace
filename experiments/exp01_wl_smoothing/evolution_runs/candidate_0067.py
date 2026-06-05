import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with exponential decay backbone."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.6

    # --- backbone: smooth exponential decay in log-space ---
    base = gamma_high * (gamma_low / gamma_high) ** progress

    # --- overflow coupling ---
    # When cells are still spread out (high overflow) we want smoother
    # gradients (higher gamma); as the layout legalizes (overflow -> 0)
    # we sharpen the WA approximation toward true HPWL.
    # Blend the time-based backbone with an overflow-driven target so the
    # schedule tracks actual placement state, not just the iteration count.
    overflow_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.2)
    w = 0.5  # equal trust in clock and physical state
    gamma = (1.0 - w) * base + w * overflow_target

    # --- plateau / divergence adaptation from HPWL trace ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            # stalled improvement -> sharpen to refine wirelength
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85
            # diverging HPWL -> smooth to recover stability
            if window[-1] > window[0] * 1.02:
                gamma *= 1.3

    # --- late-stage refinement cap ---
    if progress > 0.85:
        cap = 1.5 - 0.5 * ((progress - 0.85) / 0.15)
        gamma = min(gamma, max(gamma_low, cap))

    if gamma != gamma:  # NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))