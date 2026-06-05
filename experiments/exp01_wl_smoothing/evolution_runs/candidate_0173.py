import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma annealing with cosine fallback and late fine-tuning."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    g_high = 8.0
    g_low = 0.5

    # Primary signal: overflow. Overflow measures the true physical spread of the
    # layout (~0.9 early -> ~0.08 converged), so it tracks placement state far
    # better than raw iteration count. Map it exponentially to [g_low, g_high].
    ov_norm = (ov - 0.08) / 0.84
    ov_norm = min(1.0, max(0.0, ov_norm))
    gamma_ov = g_low * (g_high / g_low) ** ov_norm

    # Secondary signal: cosine annealing on iteration progress. Guarantees the
    # schedule keeps decaying even if overflow plateaus at a high value.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = g_high * (g_low / g_high) ** cos_prog

    # Trust overflow early (placement still spreading), blend toward the
    # progress-based decay as the run matures.
    w = 0.65 * progress
    gamma = (1.0 - w) * gamma_ov + w * gamma_prog

    # Gentle convergence feedback from HPWL trend.
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 6:
            window = recent[-5:]
            prev = recent[-6]
            best_recent = min(window)
            # Stalled: nudge toward smoother gradients to escape the plateau.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.88
            # Diverging: smooth more to restabilize.
            if window[-1] > window[0] * 1.01:
                gamma *= 1.25
            # Steadily improving: push toward accuracy.
            elif window[-1] < window[0] * 0.99:
                gamma *= 0.92

    # Late stage: clamp toward low gamma for accurate HPWL fine-tuning, but keep
    # some smoothing if the layout is still legalizing (overflow still high).
    if progress > 0.88:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.72:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = g_low
    return min(50.0, max(0.01, gamma))