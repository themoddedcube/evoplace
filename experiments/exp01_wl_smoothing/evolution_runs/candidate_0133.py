import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-gated log-cosine gamma annealing with gentle plateau adaptation."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                 # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base anneal: smooth log-space decay (cosine-shaped) high -> low ---
    # cos_prog goes 0 -> 1, slow at the ends, fast in the middle.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow gating ---
    # The schedule should not push gamma low while density is still bad:
    # blend the time-based base toward an overflow-driven target. When cells
    # are still spread out (high overflow) we keep gamma high for smooth
    # gradients; once overflow collapses we let the time schedule dominate.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.3)
    # weight on overflow term fades out as placement converges in time
    w_ov = 0.45 * (1.0 - cos_prog)
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # extra floor tied to overflow so we never go too sharp while spread out
    gamma = max(gamma, gamma_low + (gamma_high - gamma_low) * 0.5 * (ov ** 1.6))

    # --- gentle plateau / divergence adaptation ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # stalled improvement -> sharpen slightly for accuracy
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.90

            # HPWL rising (instability) -> smooth back out, but bounded
            if window[0] > 0 and window[-1] > window[0] * 1.02:
                gamma *= 1.20
            # steady descent -> ease gamma down a touch
            elif window[0] > 0 and window[-1] < window[0] * 0.98:
                gamma *= 0.97

    # --- firm late-stage caps for accurate final HPWL ---
    if progress > 0.85:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    # --- final clamp ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))