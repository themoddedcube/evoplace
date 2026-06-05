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

    it = float(iteration)

    # --- base multiplicative growth, decays toward 1.0 as placement matures ---
    # slightly slower floor than before so growth persists a touch longer
    base = max(0.99985 ** it, 0.982)

    # --- overflow-driven growth: push density weight harder while spread is poor ---
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.85)

    # --- overflow-trajectory feedback (smoothed slope of recent history) ---
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent                 # >0 means overflow is dropping

        # normalized, saturating response in (-1, 1)
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))

        if sat <= 0.0:
            # overflow stalling/rising -> push weight up more aggressively
            coef *= 1.0 - 0.062 * sat
        else:
            # overflow falling nicely -> ease off so HPWL can settle
            coef *= 1.0 - 0.050 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # --- low-overflow fine-tuning regime: relax weight so gamma can sharpen HPWL ---
    if of < 0.06:
        coef *= 0.86 + 1.05 * of
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.982

    # --- gradient-norm guard: damp growth on noisy/exploding gradients ---
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    mu = coef * base

    # --- adaptive clamp: allow gentler reduction late so HPWL can converge ---
    hi = 1.10 - 0.05 * min(max((it - 250.0) / 250.0, 0.0), 1.0)
    lo = 0.90 - 0.04 * min(max((it - 300.0) / 200.0, 0.0), 1.0)
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))