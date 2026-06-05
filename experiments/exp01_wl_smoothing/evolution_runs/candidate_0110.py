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

    # --- Base anneal: smooth log-cosine descent high -> low ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow coupling ---
    # Physical state (overflow) should dominate over the iteration clock:
    # while cells are still spread out (high overflow) keep gamma high to
    # preserve gradient signal; only let the clock pull gamma down once the
    # layout has actually condensed.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.35)
    # blend weight: trust overflow more early, trust the clock more late
    w_ov = 0.65 - 0.30 * progress
    gamma = w_ov * ov_target + (1.0 - w_ov) * base

    # gentle multiplicative nudge so high-overflow states never collapse early
    gamma *= 0.70 + 0.55 * (ov ** 1.10)

    # --- HPWL-history feedback (plateau / divergence control) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # plateau: little improvement -> sharpen toward accurate HPWL
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # diverging: HPWL climbing -> smooth gradients back out
            if window[-1] > window[0] * 1.02:
                gamma *= 1.40
            # improving steadily -> ease down a touch to refine
            elif window[-1] < window[0] * 0.97:
                gamma *= 0.92

    # --- Late-stage ceilings: force accuracy once near convergence ---
    if progress > 0.88:
        ceil = 1.3 if ov > 0.08 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.72:
        gamma = min(gamma, 2.4 if ov > 0.10 else 1.4)

    # floor early to keep gradients usable while cells still move
    if progress < 0.25 and ov > 0.5:
        gamma = max(gamma, 3.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))