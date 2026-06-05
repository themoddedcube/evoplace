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

    # slow geometric relaxation of the multiplier ceiling-floor anchor
    base = max(0.99985 ** float(iteration), 0.98)

    # overflow-proportional base coefficient (DREAMPlace-style)
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.85)

    # --- overflow-progress feedback (smoothed, asymmetric) ---
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent            # >0 means overflow falling (good)

        # soft-saturating response centered on a small positive target rate
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))          # in (-1, 1)

        if sat <= 0.0:
            # stalled / rising overflow -> push lambda harder
            coef *= 1.0 - 0.062 * sat
        else:
            # healthy descent -> ease off slightly to protect HPWL
            coef *= 1.0 - 0.050 * sat

        # second-order: detect plateau even when delta is marginally positive
        if 0.0 < delta < 1.0e-3 and of > 0.10:
            coef *= 1.018
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # --- regime-dependent damping near convergence ---
    if of < 0.06:
        # deep fine-tuning: freeze/relax lambda so HPWL can settle
        coef *= 0.86 + 1.05 * of
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.982

    # --- gradient-norm safety throttle ---
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    mu = coef * base

    # adaptive upper bound: aggressive early, gentle late for HPWL polish
    prog = min(max((float(iteration) - 250.0) / 250.0, 0.0), 1.0)
    hi = 1.10 - 0.055 * prog
    lo = 0.90 + 0.015 * prog             # raise floor late to avoid over-relaxing
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))