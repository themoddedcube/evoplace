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

    # slow geometric decay of the base growth factor; floor a touch lower so
    # the late phase can keep tightening without stalling
    base = max(0.99985 ** float(iteration), 0.975)

    # DREAMPlace-style overflow-proportional coefficient. Slightly sharper
    # exponent so high-overflow phases push harder and low-overflow phases relax.
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # trend-aware modulation: react to how fast overflow is actually dropping
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent                      # >0 means overflow improving

        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))                     # bounded in (-1, 1)

        if sat <= 0.0:
            # overflow stalled / regressing -> grow lambda faster to break plateau
            coef *= 1.0 - 0.070 * sat
        else:
            # overflow falling nicely -> let it ride, don't over-spread
            coef *= 1.0 - 0.050 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.035

    # near-convergence damping: once cells are well spread, ease lambda growth
    # so the final HPWL-tightening phase isn't disturbed by density over-push
    if of < 0.05:
        coef *= 0.86 + 1.2 * of
    elif of < 0.10:
        coef *= 0.94
    elif of < 0.18:
        coef *= 0.980

    # gradient-norm safety brake: huge gradients => unstable, grow conservatively
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.88
        elif gradient_norm > 1e4:
            coef *= 0.95

    mu = coef * base

    # iteration-dependent ceiling: allow brisk early growth, tighten the cap as
    # we age so late iterations fine-tune rather than blow density up
    prog = min(max((float(iteration) - 200.0) / 300.0, 0.0), 1.0)
    hi = 1.11 - 0.07 * prog
    lo = 0.90 - 0.02 * prog
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))