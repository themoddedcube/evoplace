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

    # base decay: gently shrink the multiplier as iterations progress,
    # with a slightly higher floor so late density push never fully dies
    base = max(0.99985 ** float(iteration), 0.985)

    # overflow-adaptive coefficient: push density harder while spread is high
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # trend term: react to the *rate* overflow is dropping
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))          # bounded in (-1, 1)
        if sat <= 0.0:
            coef *= 1.0 - 0.062 * sat     # stalling -> push harder
        else:
            coef *= 1.0 - 0.050 * sat     # progressing -> ease off, let HPWL settle
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # convergence regime: once cells are spread, prioritize HPWL accuracy
    if of < 0.06:
        coef *= 0.86 + 1.0 * of
    elif of < 0.10:
        coef *= 0.94
    elif of < 0.18:
        coef *= 0.98

    # gradient safety: damp on exploding gradients
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    mu = coef * base

    # multiplier bounds, tightening earlier and harder late for stable fine-tuning
    hi = 1.10 - 0.06 * min(max((float(iteration) - 230.0) / 250.0, 0.0), 1.0)
    mu = min(max(mu, 0.90), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))