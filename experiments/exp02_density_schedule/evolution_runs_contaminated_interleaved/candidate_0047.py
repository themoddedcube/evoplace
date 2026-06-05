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

    # slow global decay of the base multiplier: keeps lambda growing early,
    # gently relaxes the push as the layout settles
    base = max(0.99985 ** float(iteration), 0.975)

    # overflow-proportional density push (DREAMPlace WA core).
    # sub-linear exponent so the penalty stays firm while overflow is still
    # moderate, instead of collapsing toward LOWER_PCOF too early.
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # trend-aware modulation: read how fast overflow is actually dropping.
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent          # >0 means overflow improving
        # smooth saturating response centered on a small target rate
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))        # in (-1, 1)
        if sat <= 0.0:
            # stalled / worsening -> push harder to break the plateau
            coef *= 1.0 - 0.065 * sat
        else:
            # healthy progress -> ease off so HPWL is not over-penalized
            coef *= 1.0 - 0.050 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # regime-dependent damping as the placement converges.
    # once overflow is low the density term should yield to wirelength.
    if of < 0.05:
        coef *= 0.85 + 1.2 * of
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.982

    # gradient safeguard: throttle the density weight when gradients spike,
    # preventing oscillation / overshoot during the noisy low-gamma phase.
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.88
        elif gradient_norm > 1e4:
            coef *= 0.95

    mu = coef * base

    # clamp the per-iteration step. ceiling tapers down over the run so late
    # iterations cannot re-inflate lambda and undo fine wirelength tuning.
    hi = 1.10 - 0.06 * min(max((float(iteration) - 230.0) / 270.0, 0.0), 1.0)
    mu = min(max(mu, 0.90), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))