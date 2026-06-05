def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # DREAMPlace-style multiplicative density-weight growth, but made
    # overflow-adaptive and hard-clamped so it can never diverge to inf.

    LOWER, UPPER = 0.01, 50.0

    # Guard against a degenerate / uninitialized incoming weight.
    lam = current_lambda
    if not (lam == lam) or lam <= 0.0:   # NaN or non-positive
        lam = 0.1
    lam = min(max(lam, LOWER), UPPER)

    of = overflow if (overflow == overflow) else 1.0
    of = min(max(of, 0.0), 1.0)

    # Base per-iteration growth factor (>1 grows the density penalty).
    # Anneal the *cap* on growth down over iterations so early steps can
    # push hard while late steps fine-tune wirelength.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    anneal = max(0.9999 ** float(iteration), 0.98)
    base_cap = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * anneal

    # Overflow-adaptive scaling: high overflow -> grow faster to spread
    # cells; low overflow (legal-ish) -> grow gently to protect HPWL.
    # Map overflow in [0,1] to a fraction of the available growth band.
    of_drive = of ** 0.5
    mu = 1.0 + (base_cap - 1.0) * of_drive

    # Plateau escape: if overflow has stalled (not decreasing over the
    # recent window) we are stuck, so nudge the penalty up a bit more.
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-3 and of > 0.1:
            mu *= 1.02

    # Gradient safety: if gradients are exploding, ease off the growth so
    # the optimizer stays stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = lam * mu
    return float(min(max(new_lambda, LOWER), UPPER))