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

    # gentle global decay floor so updates stay multiplicative-near-1
    base = max(0.99985 ** float(iteration), 0.985)

    # overflow-adaptive base coefficient: high overflow -> push lambda up
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.85)

    # ---- overflow-velocity control (smoothed) ----
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent              # >0 means overflow is dropping (good)

        # normalize around the healthy drop rate and squash to (-1, 1)
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))

        if sat <= 0.0:
            # overflow stalled/rising -> back off harder (avoid overdriving density)
            coef *= 1.0 - 0.062 * sat
        else:
            # healthy descent -> mild boost, don't overshoot accuracy region
            coef *= 1.0 - 0.044 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # ---- terminal accuracy regime: as overflow collapses, relax lambda
    #      so gamma can drop and HPWL approximation sharpens ----
    if of < 0.05:
        coef *= 0.85 + 1.2 * of
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.982

    # ---- gradient guard: large grads -> shrink step to keep stability ----
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.89
        elif gradient_norm > 1e4:
            coef *= 0.95

    mu = coef * base

    # ---- iteration-aware clamp: allow growth early, tighten late so the
    #      schedule converges into the accurate (low-gamma) regime ----
    frac = min(max((float(iteration) - 230.0) / 270.0, 0.0), 1.0)
    hi = 1.10 - 0.06 * frac
    lo = 0.90 - 0.04 * frac
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))