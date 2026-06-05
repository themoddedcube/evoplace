def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # ---- sanitize overflow ----
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # ---- slow base decay floor so mu never collapses ----
    base = max(0.99985 ** float(iteration), 0.985)

    # ---- overflow-proportional core multiplier (DREAMPlace-style) ----
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # ---- overflow-velocity feedback: react to how fast spread is improving ----
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent                 # >0 means overflow falling (good)

        # normalized progress signal in (-1, 1)
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))

        if sat <= 0.0:
            # stalled/regressing -> push lambda harder to keep spreading
            coef *= 1.0 - 0.062 * sat
        else:
            # healthy progress -> ease off so HPWL gradient dominates
            coef *= 1.0 - 0.050 * sat

        # second-order: detect plateau even when first delta is tiny
        if len(hist) >= 6:
            mid = 0.5 * (hist[-5] + hist[-6])
            long_delta = mid - recent
            if long_delta < 2.0e-3 and of > 0.10:
                coef *= 1.025
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # ---- regime-dependent damping as placement converges ----
    if of < 0.06:
        coef *= 0.86 + 1.05 * of               # strong ease-off when nearly legal
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.982

    # ---- gradient-norm guard against divergence ----
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.88
        elif gradient_norm > 1e4:
            coef *= 0.95

    mu = coef * base

    # ---- iteration-dependent upper cap: allow fast early growth, tame late ----
    prog = min(max((float(iteration) - 220.0) / 280.0, 0.0), 1.0)
    hi = 1.11 - 0.07 * prog
    lo = 0.90 - 0.04 * prog                     # permit gentler late shrink
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))