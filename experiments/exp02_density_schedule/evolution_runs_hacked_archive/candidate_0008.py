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

    # sanitized history
    hist = [h for h in overflow_history if h == h]

    # base decay: slow, floored, so mu stays near 1 for stability
    base = max(0.9999 ** float(iteration), 0.98)

    # DREAMPlace-style overflow-driven coefficient (smooth ramp on overflow)
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # ----- plateau / progress detection -----
    # Estimate the recent rate of overflow reduction. When overflow stalls
    # at a high level we must push harder; when it is already falling fast
    # (or already low) we ease off to avoid overshoot and HPWL inflation.
    if len(hist) >= 4:
        recent = sum(hist[-2:]) / 2.0
        older = sum(hist[-4:-2]) / 2.0
        delta = older - recent  # positive => overflow decreasing
        if of > 0.20:
            # still spreading: accelerate when stalled
            if delta <= 5e-5:
                coef *= 1.06
            elif delta <= 5e-4:
                coef *= 1.025
            elif delta > 7e-3:
                coef *= 0.985
        else:
            # near convergence: bias toward gentle settling
            if delta <= 5e-5:
                coef *= 1.02
            elif delta > 4e-3:
                coef *= 0.975
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # ----- convergence-phase damping (let lambda settle for HPWL) -----
    # As overflow shrinks the placement is legal enough; further increases in
    # the density weight only distort wirelength, so taper the multiplier
    # smoothly toward (and slightly below) 1.0.
    if of < 0.05:
        coef *= 0.93
    elif of < 0.10:
        coef *= 0.955
    elif of < 0.20:
        coef *= 0.98

    # ----- gradient-norm guard -----
    # Large density gradients => step would be too aggressive; shrink growth
    # to keep the optimization on the smooth part of the WA-WL surface.
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.93
        elif gradient_norm > 1e4:
            coef *= 0.965

    # clamp the per-step multiplier to a safe band
    mu = coef * base
    mu = min(max(mu, 0.90), 1.10)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))