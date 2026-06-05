import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress annealing and stagnation control."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:        # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow
    if ov is None or ov != ov:      # None / NaN guard
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary driver: overflow ---
    # DREAMPlace-style: gamma tracks how clustered cells still are.
    # While overflow is high, keep gradients smooth; as bins clear, sharpen.
    ov_term = ov ** 1.2
    base = gamma_low + (gamma_high - gamma_low) * ov_term

    # --- secondary driver: cosine annealing on progress ---
    # Guarantees monotone-ish sharpening even if overflow plateaus.
    cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    prog_floor = gamma_low + (gamma_high - gamma_low) * cos_term

    # Blend: early iters trust progress floor, late iters trust overflow.
    w = progress
    gamma = (1.0 - w) * prog_floor + w * base
    gamma = max(gamma, gamma_low * (1.0 - progress) + 0.05 * progress)

    # --- stagnation / divergence adaptation ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            prev = recent[0]
            tail = recent[-5:]
            best_recent = min(tail)
            # stalled improvement -> sharpen to escape flat region
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.8
            # diverging HPWL -> smooth gradients to recover
            if tail[-1] > tail[0] * 1.02:
                gamma *= 1.4

    # --- late-phase fine-tuning cap ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    if gamma != gamma:              # final NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))