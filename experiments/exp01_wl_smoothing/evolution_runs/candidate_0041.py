import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware exponential gamma annealing with cosine warp and plateau handling."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base decay: blend exponential (geometric) with a cosine warp ---
    # Geometric decay keeps the schedule scale-correct across the range.
    geo = gamma_high * (gamma_low / gamma_high) ** progress
    # Cosine warp holds gamma high a touch longer, then drops smoothly to gamma_low.
    cos = gamma_low + 0.5 * (gamma_high - gamma_low) * (1.0 + math.cos(math.pi * progress))
    base = math.sqrt(max(1e-6, geo * cos))  # geometric mean of the two views

    # --- overflow coupling ---
    # When density overflow is still high, cells remain spread out, so favor
    # smoother (higher) gamma. When overflow is low, the layout is nearly legal
    # and we want accurate (lower) gamma. Keep the multiplier moderate and
    # smoothly saturating to avoid the divergence from large multipliers.
    overflow_factor = 0.75 + 1.25 * (ov ** 1.2)
    gamma = base * overflow_factor

    # --- history-driven adaptation ---
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-5:] if h is not None and h > 0]
        if len(recent) >= 3:
            first, last = recent[0], recent[-1]
            best_recent = min(recent)
            # Plateau: HPWL barely improving -> sharpen (lower gamma) to refine.
            if first > 0 and (first - best_recent) / first < 1e-3:
                gamma *= 0.75
            # Diverging: HPWL climbing -> smooth (raise gamma) to recover stability.
            if last > first * 1.01:
                gamma *= 1.3

    # --- late-stage accuracy guard (soft, progress-scaled) ---
    if progress > 0.8:
        cap = gamma_low + (1.5 - gamma_low) * (1.0 - progress) / 0.2
        gamma = min(gamma, max(gamma_low, cap))

    return min(50.0, max(0.01, gamma))