import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma annealing with progress guards and plateau adaptation.

    Primary driver is overflow (the true measure of placement convergence in
    DREAMPlace): high overflow -> high gamma (smooth gradients while cells are
    still spreading), low overflow -> low gamma (accurate HPWL for fine-tuning).
    Iteration progress acts only as a safety floor/ceiling so a stalled overflow
    signal cannot keep gamma pinned high forever.
    """

    GAMMA_HIGH = 8.0
    GAMMA_LOW = 0.5
    GAMMA_FLOOR = 0.01
    GAMMA_CEIL = 50.0

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    # --- overflow-driven geometric anneal --------------------------------
    # f(ov) in [0,1]: 1 when fully overflowing, 0 when density satisfied.
    # sqrt-shaping holds gamma higher across the mid-overflow regime where
    # most legalization-relevant spreading happens.
    f_ov = math.sqrt(ov)
    gamma_ov = GAMMA_HIGH * (GAMMA_LOW / GAMMA_HIGH) ** (1.0 - f_ov)

    # --- iteration cosine anneal (secondary driver) ----------------------
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_it = GAMMA_HIGH * (GAMMA_LOW / GAMMA_HIGH) ** cos_prog

    # Blend: trust overflow most early/mid, lean on iteration schedule late
    # so the run reliably reaches accurate-HPWL territory by the end.
    w_it = 0.30 + 0.40 * progress
    gamma = (1.0 - w_it) * gamma_ov + w_it * gamma_it

    # --- plateau / divergence adaptation from HPWL history ----------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # HPWL diverging -> gradients too noisy, smooth them out.
            if last > first * 1.02:
                gamma *= 1.30
            # Stalled improvement -> sharpen toward accurate HPWL.
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85
            # Healthy steady descent -> nudge slightly sharper.
            elif last < first * 0.98:
                gamma *= 0.96

    # --- end-game ceilings so we always fine-tune ------------------------
    if progress > 0.85:
        gamma = min(gamma, 1.3 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:  # NaN guard
        gamma = GAMMA_LOW
    return min(GAMMA_CEIL, max(GAMMA_FLOOR, gamma))