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

    # slow global damping so growth tapers as the run matures
    base = max(0.99985 ** float(iteration), 0.985)

    # overflow-adaptive center: push lambda up hard while many bins are over
    # density, ease off smoothly as the layout legalizes
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # ---- overflow-trend control --------------------------------------
    # When overflow stops improving, the density force is already strong
    # enough; back off the multiplier so HPWL is not sacrificed.
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent                 # >0 means improving
        # smooth saturating response in [-1, 1]
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))
        if sat <= 0.0:
            # stalling / regressing -> relax lambda growth more
            coef *= 1.0 - 0.062 * sat
        else:
            # healthy progress -> only mild boost, keep wirelength low
            coef *= 1.0 - 0.040 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # ---- low-overflow fine-tuning ------------------------------------
    # Near convergence, stop inflating lambda so the optimizer can pull
    # wirelength back in without disturbing legality.
    if of < 0.05:
        coef *= 0.86 + 1.2 * of
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.982

    # ---- gradient-norm safety ----------------------------------------
    # Large gradients mean the step is already aggressive; temper lambda
    # to avoid overshoot and oscillation.
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    mu = coef * base

    # adaptive upper cap: allow brisk early growth, tighten late so the
    # endgame favors wirelength refinement over further densification
    hi = 1.10 - 0.06 * min(max((float(iteration) - 230.0) / 270.0, 0.0), 1.0)
    mu = min(max(mu, 0.90), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))