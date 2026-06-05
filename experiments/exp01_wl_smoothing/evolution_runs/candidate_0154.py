import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule for WA-WL smoothing in differentiable
    global placement. Density (overflow) is the primary signal of true
    convergence, so it governs gamma; iteration progress only nudges, and the
    floor is tied to overflow so gamma never collapses while bins are still
    over-dense (which would inject noise and diverge HPWL)."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: exponential in overflow (DREAMPlace-style).
    # overflow ~1 -> high gamma (smooth gradients while cells cluster);
    # overflow ~0 -> low gamma (accurate HPWL for fine-tuning).
    ov_term = gamma_low * (gamma_high / gamma_low) ** (ov ** 0.85)

    # Secondary: gentle cosine progress decay so we keep sharpening even if
    # overflow lingers, but it never forces gamma to collapse on its own.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_term = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend, weighting overflow more heavily since it tracks real convergence.
    gamma = 0.70 * ov_term + 0.30 * prog_term

    # Plateau / divergence adaptation from the HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # HPWL climbing -> gradients too noisy, smooth more.
            if last > first * 1.01:
                gamma *= 1.25
            # Plateaued and not worsening -> sharpen to refine the solution.
            elif prev > 0 and (prev - best) / prev < 1e-3 and last <= first:
                gamma *= 0.90

    # Floor tied to overflow, NOT raw progress: only permit very low gamma once
    # density is genuinely resolved; otherwise keep gradients smooth.
    if ov < 0.08:
        gamma = max(gamma, 0.05)
    else:
        gamma = max(gamma, 0.30 + 2.0 * ov)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))