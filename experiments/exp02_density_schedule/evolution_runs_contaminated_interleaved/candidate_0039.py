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

    # slow geometric base decay, floored
    base = max(0.99985 ** float(iteration), 0.98)

    # overflow-adaptive coefficient: push harder while spread is high,
    # ease off smoothly as bins clear out
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # trend-aware modulation from overflow history
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent            # positive => overflow dropping
        # smooth, bounded response (soft-sign)
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))
        if sat <= 0.0:
            # overflow stalling/rising -> push penalty up more
            coef *= 1.0 - 0.062 * sat
        else:
            # healthy progress -> relax penalty growth gently
            coef *= 1.0 - 0.048 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # late-stage fine-tuning: as overflow gets low, stop inflating lambda
    if of < 0.05:
        coef *= 0.86 + 1.2 * of
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.982

    # damp updates when gradients blow up (noisy/unstable region)
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.89
        elif gradient_norm > 1e4:
            coef *= 0.95

    mu = coef * base

    # adaptive upper clamp: generous early, tighter once converging
    prog = min(max((float(iteration) - 240.0) / 260.0, 0.0), 1.0)
    hi = 1.10 - 0.055 * prog
    mu = min(max(mu, 0.90), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))