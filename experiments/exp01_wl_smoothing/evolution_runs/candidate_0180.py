import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log schedule with gentle iteration-based annealing.

    Primary signal is overflow (proven DREAMPlace behavior): high overflow ->
    smooth high gamma for clustering; low overflow -> low gamma for accurate
    HPWL. A cosine iteration term acts as a soft budget so fine-tuning still
    happens if overflow plateaus, and HPWL trend nudges stability/accuracy.
    """

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: log-linear in overflow.
    # ov=1 (heavy overlap) -> gamma_high (smooth); ov=0 (legal) -> gamma_low.
    ov_shaped = ov ** 0.85
    gamma_ov = gamma_low * (gamma_high / gamma_low) ** ov_shaped

    # Secondary driver: cosine annealing in iteration as a soft ceiling so
    # late iterations still fine-tune even if overflow stalls high.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_iter = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: mostly overflow-driven, never far above the iteration budget.
    gamma = 0.7 * gamma_ov + 0.3 * gamma_iter
    gamma = min(gamma, 1.5 * gamma_iter)

    # HPWL-trend response (gentle, to avoid the divergence that crushing caused).
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 6:
            w = recent[-5:]
            if w[-1] > w[0] * 1.01:          # rising -> re-smooth for stability
                gamma *= 1.20
            elif w[-1] < w[0] * 0.995:        # steadily improving -> sharpen a bit
                gamma *= 0.92

    # Late-stage soft floor toward accurate HPWL, but overflow-aware so we do
    # not force low gamma while cells still overlap (which diverges).
    if progress > 0.85:
        cap = 2.0 if ov > 0.10 else 1.0
        gamma = min(gamma, cap)
    elif progress > 0.70:
        cap = 3.0 if ov > 0.10 else 1.8
        gamma = min(gamma, cap)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))