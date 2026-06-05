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

    # mild global decay so step size shrinks as placement matures
    base = max(0.99985 ** float(iteration), 0.98)

    # DREAMPlace-style overflow-adaptive coefficient: push harder while
    # spreading is incomplete, relax as bins clear.
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.85)

    # ---- overflow-trend adaptation -------------------------------------
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent            # >0 means overflow is dropping

        # smooth saturating response in [-1, 1]
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))

        if sat <= 0.0:
            # overflow stalled / rising -> push lambda up to keep spreading
            coef *= 1.0 - 0.060 * sat
        else:
            # overflow falling nicely -> ease off so HPWL can settle
            coef *= 1.0 - 0.048 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # ---- regime-dependent damping near convergence ---------------------
    if of < 0.05:
        # nearly legal: strongly damp lambda growth, let gamma/HPWL refine
        coef *= 0.85 + 1.2 * of
    elif of < 0.10:
        coef *= 0.94
    elif of < 0.18:
        coef *= 0.982

    # ---- gradient-norm safeguard ---------------------------------------
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.88
        elif gradient_norm > 1e4:
            coef *= 0.95

    mu = coef * base

    # ---- clamp multiplier: allow brisk early growth, tighten the ceiling
    #      over time so late iterations cannot over-inflate lambda --------
    prog = min(max((float(iteration) - 230.0) / 270.0, 0.0), 1.0)
    hi = 1.10 - 0.06 * prog
    lo = 0.90 + 0.02 * prog
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))