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

    # mild global decay floor, relaxed a touch lower so mature placements
    # bleed off density pressure and fine-tune wirelength instead of holding
    base = max(0.99980 ** float(iteration), 0.975)

    # overflow-magnitude coefficient (DREAMPlace-style)
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.85)

    # overflow-trend control: smooth saturating response to descent rate.
    # The healthy-descent center is nudged up and the stall-boost softened so
    # the self-regulating equilibrium parks at marginally higher overflow
    # (~0.353) with less lambda pressure -- the regime that minimized HPWL.
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent            # positive == overflow decreasing
        x = (delta - 1.2e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))          # in (-1, 1)
        if sat <= 0.0:
            coef *= 1.0 - 0.050 * sat     # gentler push when stalled
        else:
            coef *= 1.0 - 0.048 * sat     # ease slightly more when collapsing
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.025

    # near-legal regime: damp lambda growth so wirelength can be refined
    if of < 0.06:
        coef *= 0.88 + 1.0 * of
    elif of < 0.10:
        coef *= 0.95
    elif of < 0.18:
        coef *= 0.985

    # gradient safeguard
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    mu = coef * base

    # iteration-aware growth ceiling: tightened earlier and lower so late
    # density pushes cannot over-spread cells and inflate HPWL.
    hi = 1.09 - 0.05 * min(max((float(iteration) - 200.0) / 250.0, 0.0), 1.0)
    mu = min(max(mu, 0.90), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))