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
    if progress != progress:          # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base geometric decay over progress ---
    base = gamma_high * (gamma_low / gamma_high) ** progress

    # --- overflow drives the schedule more than raw iteration count ---
    # When cells are still spread out (high overflow) we want smooth, high gamma.
    # As the layout legalizes (overflow -> 0) we sharpen toward accurate HPWL.
    # Blend the time-based base with an overflow-based target so a fast-
    # converging run anneals early instead of waiting for the iteration count.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 0.85)
    blend = 0.5 + 0.5 * progress          # trust overflow early, schedule late
    gamma = (1.0 - blend) * ov_target + blend * base

    # mild multiplicative overflow nudge to keep gradients smooth while packed
    gamma *= 0.75 + 0.45 * ov

    # --- adapt to HPWL trajectory ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[0]
            best_recent = min(window)
            # stagnation: sharpen to chase a better optimum
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.7
            # divergence/oscillation: smooth back out for stability
            if window[-1] > min(window) * 1.03:
                gamma *= 1.4

    # --- late-stage cap: prioritize accurate HPWL once nearly legal ---
    if progress > 0.9 or ov < 0.08:
        gamma = min(gamma, 0.9)
    elif progress > 0.8:
        gamma = min(gamma, 1.5)

    if gamma != gamma:                # final NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))