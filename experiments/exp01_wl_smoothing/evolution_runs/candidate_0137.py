import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-led gamma schedule: high γ while cells are clustered,
    decaying to low γ for accurate HPWL fine-tuning, with a cosine
    progress fallback and HPWL-trend correction."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Overflow is the most reliable convergence signal in DREAMPlace.
    # Above the target overflow -> elevated gamma; below -> collapse to low gamma.
    target_ov = 0.08
    ov_factor = (ov - target_ov) / (1.0 - target_ov)
    ov_factor = min(1.0, max(0.0, ov_factor))
    ov_factor = ov_factor ** 0.75          # keep gamma up while overflow is moderate

    # Smooth cosine fallback (high early -> low late) so gamma never collapses
    # prematurely before density information is meaningful.
    prog_factor = 0.5 + 0.5 * math.cos(math.pi * progress)

    # Overflow leads; progress guards the early phase.
    f = 0.7 * ov_factor + 0.3 * prog_factor
    f = min(1.0, max(0.0, f))

    gamma = gamma_low * (gamma_high / gamma_low) ** f

    # HPWL-trend correction over the recent window.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            if window[-1] > window[0] * 1.02:      # rising -> too noisy, smooth more
                gamma *= 1.30
            elif window[-1] < window[0] * 0.98:    # descending well -> push accuracy
                gamma *= 0.92
            else:                                  # plateau -> sharpen for fine-tuning
                gamma *= 0.85

    # Late-stage caps to guarantee accurate HPWL once near-legal.
    if progress > 0.85:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.2)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))