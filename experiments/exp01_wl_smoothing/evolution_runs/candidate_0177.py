import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with cosine progress annealing and
    HPWL feedback. gamma tracks overflow geometrically in log-space:
    high overflow -> smooth (high gamma), low overflow -> accurate (low gamma)."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: overflow-adaptive geometric interpolation in log-space.
    # This is the proven DREAMPlace-style coupling — gamma follows the spread
    # of the layout rather than a fixed clock, which keeps gradients smooth
    # exactly while cells are still overlapping and sharpens as they settle.
    # sqrt warps the curve so gamma drops meaningfully as overflow falls.
    t = ov ** 0.5
    base = gamma_low * (gamma_high / gamma_low) ** t

    # Secondary: gentle cosine annealing guarantees continued fine-tuning even
    # if overflow plateaus, without dominating the overflow signal.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma = base * (1.0 - 0.35 * cos_prog)

    # HPWL feedback: react to stagnation and divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # improvement stalled -> sharpen for a more accurate HPWL
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.9

            # HPWL climbing (instability) -> smooth gradients to recover
            if window[-1] > window[0] * 1.02:
                gamma *= 1.3
            # HPWL falling steadily -> let it keep sharpening slightly
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.97

    # Late-stage ceilings: force accurate final HPWL, but only once the layout
    # is sufficiently legal (low overflow) so we never crush a still-overlapping
    # placement into noisy gradients.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))