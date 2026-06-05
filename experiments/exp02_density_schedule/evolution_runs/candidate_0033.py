def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # sanitize overflow
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # slow global decay of the update ceiling (cells settle over time)
    base = max(0.99985 ** float(iteration), 0.975)

    # DREAMPlace-style overflow-proportional growth of the penalty multiplier
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.85)

    # ---- progress-aware modulation from overflow history ----
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent                 # >0 means overflow still dropping

        # smooth saturating response to the rate of overflow reduction
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))               # in (-1, 1)

        if sat <= 0.0:
            # overflow stalled/rising -> back off harder to let HPWL relax
            coef *= 1.0 - 0.062 * sat
        else:
            # healthy progress -> let penalty keep climbing, but gently
            coef *= 1.0 - 0.044 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # ---- regime control by absolute overflow level ----
    # near-converged: aggressively damp so the last iters refine HPWL, not spread
    if of < 0.06:
        coef *= 0.86 + 1.1 * of
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.982

    # ---- gradient-norm safety brake (avoid penalty blow-up) ----
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.89
        elif gradient_norm > 1e4:
            coef *= 0.95

    mu = coef * base

    # adaptive ceiling: generous early (force clustering), tight late (settle HPWL)
    prog = min(max((float(iteration) - 230.0) / 260.0, 0.0), 1.0)
    hi = 1.10 - 0.06 * prog
    lo = 0.90 + 0.02 * prog
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))