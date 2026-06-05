import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with a progress floor and plateau adaptation.

    Primary signal is overflow (a faithful proxy for placement spread); a
    cosine-annealed progress term acts as a guaranteed pull toward the
    accurate-HPWL (low gamma) regime late in the run. HPWL history nudges
    gamma down on stagnation and up on divergence.
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

    # Primary signal: log-interpolate gamma_low (ov->0) .. gamma_high (ov->1).
    ov_log = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov)

    # Secondary signal: cosine-annealed progress, a fallback in case overflow
    # plateaus far from zero (it often does on hard designs).
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_log = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Trust overflow most early; let progress pull gamma down as the run ends,
    # so we always reach the accurate-HPWL regime even if overflow lingers.
    w_prog = 0.30 + 0.40 * progress
    gamma = (1.0 - w_prog) * ov_log + w_prog * min(ov_log, prog_log)

    # Plateau / divergence adaptation from HPWL history.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Stagnation: sharpen the approximation to chase real HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence vs. steady improvement.
            if last > first * 1.02:
                gamma *= 1.35
            elif last < first * 0.98:
                gamma *= 0.95

    # Late-stage ceilings to guarantee fine-tuning, relaxed if still spread.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))