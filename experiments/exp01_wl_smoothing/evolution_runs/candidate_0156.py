import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-gated log-cosine gamma schedule with plateau adaptation."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth log-space decay (cosine ease) from high -> low over progress.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow gate: cells still spreading (high ov) -> keep gamma high for
    # stable, smooth gradients; well-spread (low ov) -> allow accurate low gamma.
    # This is the primary safeguard against divergence (inf HPWL) when gamma
    # drops too early while density is still poor.
    ov_floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    gamma = max(base, 0.5 * base + 0.5 * ov_floor)

    # Plateau / oscillation adaptation from HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0 and h != float("inf")]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stagnation: tighten gamma to refine wirelength.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence guard: HPWL climbing -> raise gamma to re-smooth.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.40
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # Late-stage ceilings, but never crush gamma while density is still high.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # Hard safety floor when overflow is severe to prevent gradient noise blowup.
    if ov > 0.45:
        gamma = max(gamma, 2.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))