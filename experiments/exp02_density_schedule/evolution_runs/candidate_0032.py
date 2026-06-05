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

    # base power decay (mirrors DREAMPlace iteration damping); floor a touch higher
    base = max(0.99985 ** float(iteration), 0.985)

    # overflow-driven coefficient: high overflow -> push lambda up to spread cells
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # overflow-trend response: plateau/regression -> ramp harder; fast drop -> ease
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent            # >0 == overflow improving
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))          # in (-1, 1)
        if sat <= 0.0:
            coef *= 1.0 - 0.065 * sat     # plateau/regress -> increase
        else:
            coef *= 1.0 - 0.040 * sat     # improving -> mild ease
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # convergence regime: ease lambda as overflow becomes small (protect HPWL)
    if of < 0.05:
        coef *= 0.86 + 1.2 * of
    elif of < 0.10:
        coef *= 0.94
    elif of < 0.18:
        coef *= 0.98

    # gradient-norm safety damping (gentler than before to avoid over-suppression)
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.92
        elif gradient_norm > 1e4:
            coef *= 0.96

    mu = coef * base

    # multiplier caps: allow stronger early ramp, tighten window late
    frac = min(max((float(iteration) - 200.0) / 300.0, 0.0), 1.0)
    hi = 1.12 - 0.06 * frac
    lo = 0.90 + 0.03 * frac
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))