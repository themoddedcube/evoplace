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

    # base annealing: slow geometric drift, capped at 0.98
    base = max(0.9999 ** float(iteration), 0.98)

    # overflow-proportional coefficient: high overflow -> push density weight up harder
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # smoothed overflow trend over a short window (robust to single-step noise)
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = sum(hist[-2:]) / 2.0
        older = sum(hist[-4:-2]) / 2.0
        delta = older - recent  # positive = still improving
        if delta <= 5e-5:
            coef *= 1.05          # firmly stalled: accelerate legalization
        elif delta <= 5e-4:
            coef *= 1.02          # slow progress: nudge up
        elif delta > 6e-3:
            coef *= 0.98          # fast progress: ease off to protect HPWL
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # late-stage wirelength refinement: once nearly legal, relax weight so the
    # accurate (low-gamma) HPWL gradient can sharpen the solution
    if of < 0.10:
        coef *= 0.96
    elif of < 0.20:
        coef *= 0.985

    # gradient guard: damp blow-ups from noisy low-gamma gradients
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 1e4:
            coef *= 0.97
        elif gradient_norm > 5e4:
            coef *= 0.94

    mu = coef * base
    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))