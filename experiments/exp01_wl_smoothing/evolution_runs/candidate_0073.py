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
    if progress < 0.0:
        progress = 0.0
    elif progress > 1.0:
        progress = 1.0

    ov = overflow if overflow is not None else 1.0
    if ov != ov:  # NaN guard
        ov = 1.0
    if ov < 0.0:
        ov = 0.0
    elif ov > 1.0:
        ov = 1.0

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base anneal: blend exponential decay with a cosine ramp ---
    # exponential keeps geometric spacing; cosine holds gamma high a bit
    # longer early then drops smoothly into fine-tuning.
    exp_decay = gamma_high * (gamma_low / gamma_high) ** progress
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    cos_anneal = gamma_low + (gamma_high - gamma_low) * cos_factor
    base = 0.5 * exp_decay + 0.5 * cos_anneal

    # --- overflow adaptation ---
    # high overflow => cells still spread => want smoother (higher) gamma.
    # low overflow => mostly legalized => sharpen (lower) gamma.
    overflow_factor = 0.55 + 1.85 * (ov ** 1.3)
    gamma = base * overflow_factor

    # --- history-driven feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        finite_recent = [h for h in recent if h == h and abs(h) != float("inf")]
        if len(finite_recent) >= 2:
            prev = hpwl_history[-6] if len(hpwl_history) >= 6 else finite_recent[0]
            best_recent = min(finite_recent)
            # stagnation: relative improvement very small -> sharpen to escape plateau
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75
            # divergence: HPWL trending up -> smooth gradients to restabilize
            if finite_recent[-1] > finite_recent[0] * 1.02:
                gamma *= 1.4

    # --- late-stage cap for accurate HPWL ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    # --- final clamp ---
    if gamma != gamma:  # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))