import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress-based decay floor."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary signal: overflow. DREAMPlace-style log-space interpolation.
    # Overflow typically sweeps from ~1.0 (start) down toward ~0.1 (converged).
    ov_lo, ov_hi = 0.10, 0.95
    t = (ov - ov_lo) / (ov_hi - ov_lo)
    t = min(1.0, max(0.0, t))
    gamma_ov = gamma_low * (gamma_high / gamma_low) ** t

    # Backstop signal: monotone cosine decay on iteration progress, so gamma
    # still relaxes even if overflow plateaus high (avoids stuck-smooth blowups).
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Trust overflow early; in the back half, never exceed the progress backstop.
    w = progress
    gamma = (1.0 - w) * gamma_ov + w * min(gamma_ov, gamma_prog)

    # Adaptive response to the HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-7:] if h is not None and h == h and h > 0]
        if len(recent) >= 6:
            window = recent[-5:]
            prev = recent[-6]
            best_recent = min(window)

            # Stalled improvement -> sharpen (lower gamma) for finer HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.9

            # Diverging HPWL -> smooth (raise gamma) to stabilize gradients.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.2

    # Late-stage fine-tuning cap pushes toward accurate HPWL once near-legal.
    if progress > 0.85:
        ceil = 1.2 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)

    return min(50.0, max(0.01, gamma))