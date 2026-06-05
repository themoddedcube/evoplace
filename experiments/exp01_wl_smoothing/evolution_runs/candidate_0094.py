import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with a cosine-annealed progress backstop
    and HPWL-trend correction. High gamma while density overflow is large
    (smooth gradients, cells cluster), low gamma as the placement legalizes
    (accurate HPWL, fine-tuning)."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: overflow. Geometric interpolation in log-gamma space so
    # gamma decays smoothly from high->low as the layout legalizes. The ov**0.8
    # exponent keeps gamma elevated a bit longer while overflow is still high.
    ov_factor = ov ** 0.8
    gamma_ov = gamma_low * (gamma_high / gamma_low) ** ov_factor

    # Backstop: cosine annealing on iteration progress guarantees a monotone
    # descent even if overflow plateaus or stalls.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow dominates; progress contributes a steady descent floor.
    gamma = 0.70 * gamma_ov + 0.30 * gamma_prog

    # HPWL-trend correction over a short clean window.
    if hpwl_history:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 4:
            first = recent[0]
            last = recent[-1]
            best = min(recent)
            if last > first * 1.01:                 # diverging -> smooth more
                gamma *= 1.25
            elif (first - best) / first < 1e-3:     # plateau -> sharpen
                gamma *= 0.85
            elif last < first * 0.98:               # improving -> ease down
                gamma *= 0.95

    # Late-phase ceilings: once near-legal, force accurate (low) gamma to lock
    # in true HPWL; relax the cap if overflow is still meaningful.
    if progress > 0.85:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.2)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))