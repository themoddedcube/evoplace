import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma annealing for WA-WL global placement.

    High gamma early (smooth gradients, cells cluster) -> low gamma late
    (accurate HPWL, fine placement). Decay is primarily driven by the live
    overflow signal and gated by iteration progress, with mild plateau /
    divergence reactions from the HPWL trace.
    """

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    if ov != ov:                       # NaN guard
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base schedule: blend of overflow- and progress-driven decay -----
    # Overflow is the physically meaningful signal: while bins are packed
    # we want smooth gradients; as density relaxes we sharpen the WL model.
    # We map overflow through a smooth curve so early iterations (ov~1) stay
    # near gamma_high and the tail (ov->0) approaches gamma_low.
    ov_curve = ov ** 0.85
    gamma_ov = gamma_low + (gamma_high - gamma_low) * ov_curve

    # Exponential (geometric) decay on progress as a fallback / floor so the
    # schedule keeps cooling even if overflow stalls.
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** progress

    # Take the gentler-cooling of the two early on, but let progress force
    # convergence late: weight shifts toward the progress schedule over time.
    w = progress
    gamma = (1.0 - w) * gamma_ov + w * min(gamma_ov, gamma_prog)

    # --- HPWL-trace feedback (robust to short / dirty histories) ---------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-5:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 3:
            first = recent[0]
            last = recent[-1]
            best_recent = min(recent)
            prev = None
            if len(hpwl_history) >= 6:
                p = hpwl_history[-6]
                if p is not None and p == p and p > 0:
                    prev = p
            if prev is None:
                prev = first

            # Plateau: little improvement -> sharpen to escape via accuracy.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> re-smooth gradients to stabilize.
            if last > first * 1.02:
                gamma *= 1.3

    # --- late-stage clamp: guarantee accurate WL at convergence ----------
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.5)

    # --- final guard -----------------------------------------------------
    if gamma != gamma:                 # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, float(gamma)))