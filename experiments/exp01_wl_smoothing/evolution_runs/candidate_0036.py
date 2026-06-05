import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with smooth progress decay and
    convergence-aware fine-tuning for differentiable global placement."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    if ov != ov:  # NaN guard
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary driver: overflow ---
    # DREAMPlace-style coupling: gamma tracks how spread-out the cells still are.
    # While overflow is high (cells overlapping) keep gradients smooth; as the
    # layout legalizes (overflow -> 0) sharpen toward an accurate HPWL.
    ov_term = ov ** 1.2
    gamma_ov = gamma_low + (gamma_high - gamma_low) * ov_term

    # --- secondary driver: scheduled progress (exponential anneal) ---
    # Guarantees monotone sharpening even if overflow plateaus.
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** progress

    # Blend: early on let progress hold gamma high; later let overflow dominate
    # so well-legalized regions get accurate gradients sooner.
    w = progress  # shift weight from schedule -> overflow over time
    gamma = (1.0 - w) * gamma_prog + w * gamma_ov

    # Keep some smoothing floor early to avoid noisy gradients before cells settle.
    early_floor = gamma_low + (gamma_high - gamma_low) * (1.0 - progress) ** 2 * ov
    gamma = max(gamma, early_floor)

    # --- convergence-aware adaptation from HPWL trend ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-5:] if h is not None and h == h and h > 0]
        if len(recent) >= 3:
            prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
            best_recent = min(recent)

            # Stagnation: HPWL barely improving -> sharpen to refine placement.
            if prev and prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.8

            # Divergence: HPWL climbing -> re-smooth gradients to recover.
            if recent[-1] > recent[0] * 1.02:
                gamma *= 1.3
                gamma = min(gamma, gamma_high)

    # --- late-stage accuracy clamp ---
    # Near the end, force low gamma for an accurate final HPWL.
    if progress > 0.9:
        gamma = min(gamma, 1.0)
    elif progress > 0.8:
        gamma = min(gamma, 2.0)

    if gamma != gamma:  # final NaN guard
        gamma = 1.0

    return min(50.0, max(0.01, gamma))