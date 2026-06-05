def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # --- sanitize inputs ---
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    hist = [h for h in overflow_history if h == h]

    # --- annealing envelope: density weight grows fast early, settles late ---
    # slow geometric decay of the per-step growth ceiling so late iters fine-tune
    base = max(0.99980 ** float(iteration), 0.975)

    # --- overflow-magnitude term ---
    # large overflow => spreading is far from done => grow lambda toward UPPER.
    # small overflow => near-legal => damp growth so wirelength can relax.
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # --- overflow-trend term (proxy for spreading progress) ---
    # use a smoothed slope of recent overflow; stalled/rising overflow => push harder.
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent  # >0 means overflow is dropping (good)
        if delta <= 0.0:
            coef *= 1.07          # stalled or worsening: push spreading
        elif delta <= 5e-5:
            coef *= 1.05
        elif delta <= 5e-4:
            coef *= 1.02
        elif delta > 1.5e-2:
            coef *= 0.95          # dropping very fast: ease off, let WL settle
        elif delta > 8e-3:
            coef *= 0.975
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.04

    # --- late-stage damping by overflow level ---
    if of < 0.05:
        coef *= 0.85 + 1.2 * of   # strong damp once nearly legalizable
    elif of < 0.10:
        coef *= 0.945
    elif of < 0.18:
        coef *= 0.98

    # --- gradient-norm safeguard: huge gradients => slow lambda growth ---
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.88
        elif gradient_norm > 1e4:
            coef *= 0.945

    mu = coef * base
    mu = min(max(mu, 0.88), 1.10)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))