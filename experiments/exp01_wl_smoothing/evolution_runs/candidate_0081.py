import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-decayed gamma schedule for WA-WL placement."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base time schedule: smooth log-cosine decay high -> low ---
    # cos_prog ramps 0 -> 1 with gentle ends, giving a geometric interpolation
    # in log-space so early iterations stay smooth and late ones get accurate.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # When the layout is still spread out (high overflow) we want smoother
    # gradients, so lift gamma; as bins clear we let the time schedule dominate.
    # Blend multiplicatively but bounded to avoid runaway values.
    ov_lift = 0.6 + 1.6 * (ov ** 1.5)          # in [0.6, 2.2]
    ov_floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.2)

    gamma = 0.5 * base * ov_lift + 0.5 * ov_floor

    # --- HPWL feedback (defensive parsing) ---
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and 0.0 < h < float("inf")]
        if len(recent) >= 4:
            window = recent[-4:]
            best_recent = min(window)
            prev = recent[-5] if len(recent) >= 5 else window[0]

            # plateau: HPWL barely improving -> sharpen (lower gamma) to refine
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # divergence: HPWL climbing -> smooth (raise gamma) to restabilize
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25

    # --- late-stage annealing cap: force accurate HPWL near the end ---
    if progress > 0.85:
        ceil = 1.2 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)

    # --- final clamp ---
    if gamma != gamma:   # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))