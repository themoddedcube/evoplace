import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-space gamma schedule with progress annealing."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver (DREAMPlace-style): gamma stays high while bins are
    # congested and decays smoothly toward gamma_low in log-space as the
    # placement legalizes. sqrt emphasizes moving low once overflow drops.
    ov_blend = ov ** 0.5
    g_ov = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_blend)

    # Secondary driver: cosine annealing on iteration progress (log-space).
    # Acts as a monotone backstop so gamma still descends even if the
    # overflow signal is noisy or plateaus early.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    g_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Trust overflow early (cells still spreading), lean on the progress
    # backstop late so gamma is never pulled back up near the end.
    w = progress
    gamma = (1.0 - w) * g_ov + w * min(g_ov, g_prog)

    # Mild plateau response: if HPWL has stalled, sharpen the approximation
    # by nudging gamma down. Bounded, single multiplier — no blow-up.
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-6:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 6:
            prev = recent[0]
            best_rest = min(recent[1:])
            if prev > 0 and (prev - best_rest) / prev < 1e-3:
                gamma *= 0.90
            # If HPWL is actively worsening, gradients may be too noisy:
            # allow a small, capped increase for smoother optimization.
            elif recent[-1] > recent[0] * 1.02:
                gamma *= 1.15

    # Final-phase caps: once nearly legal, force accurate HPWL.
    if progress > 0.90:
        gamma = min(gamma, 0.7 if ov < 0.10 else 1.5)
    elif progress > 0.75:
        gamma = min(gamma, 1.5 if ov < 0.10 else 2.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))