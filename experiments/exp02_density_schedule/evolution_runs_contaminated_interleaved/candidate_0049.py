def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # sanitize overflow (NaN-safe), clamp to [0, 1]
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # slow geometric decay of the base growth factor; floor kept low so the
    # late phase keeps tightening wirelength instead of stalling
    base = max(0.99985 ** float(iteration), 0.975)

    # DREAMPlace-style overflow-proportional density push (WA core).
    # sub-linear exponent keeps the penalty firm at moderate overflow.
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # trend-aware modulation: respond to how fast overflow is actually dropping.
    # Center the saturating response so the self-regulating equilibrium parks
    # near the HPWL-optimal overflow band (~0.353).
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent                 # >0 means overflow improving
        x = (delta - 1.1e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))                # bounded in (-1, 1)
        if sat <= 0.0:
            # stalled / regressing -> grow lambda to break the plateau
            coef *= 1.0 - 0.060 * sat
        else:
            # healthy descent -> ease off so HPWL is not over-penalized
            coef *= 1.0 - 0.052 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.030

    # equilibrium-band hold: when overflow sits in the HPWL-optimal window,
    # damp net lambda growth so the layout settles rather than oscillates.
    if of < 0.05:
        coef *= 0.86 + 1.2 * of
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.980
    elif of < 0.40:
        coef *= 0.992

    # gradient-norm safety brake: throttle density weight when gradients spike
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.88
        elif gradient_norm > 1e4:
            coef *= 0.95

    mu = coef * base

    # iteration-aware ceiling: brisk early growth, tightening cap as the run
    # ages so late iterations fine-tune instead of re-inflating density
    prog = min(max((float(iteration) - 220.0) / 280.0, 0.0), 1.0)
    hi = 1.10 - 0.07 * prog
    lo = 0.90 - 0.02 * prog
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))