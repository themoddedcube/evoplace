def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight (lambda) schedule with hard clamping.

    Grows lambda geometrically while bins remain congested, but modulates the
    growth rate by how fast overflow is actually falling and by the gradient
    norm, then clamps the result to the legal range so it can never diverge.
    """
    LO, HI = 0.01, 50.0

    # Sanitize inputs (guard against NaN/inf coming back from the solver).
    if not (current_lambda == current_lambda) or current_lambda in (float("inf"), float("-inf")):
        current_lambda = 1.0
    current_lambda = min(max(current_lambda, LO), HI)

    of = overflow if (overflow == overflow) else 1.0
    of = min(max(of, 0.0), 1.0)

    # Base multiplier: DREAMPlace-style mild geometric increase, annealed by iter.
    UPPER_PCOF, LOWER_PCOF = 1.05, 1.003
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow shapes the multiplier: congested -> push harder, converged -> ease off.
    # of high  -> mu near UPPER_PCOF ; of low -> mu near LOWER_PCOF.
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * decay * of

    # Stagnation check: if overflow stopped improving, nudge lambda up to break ties.
    if overflow_history and len(overflow_history) >= 3:
        recent = [h for h in overflow_history[-3:] if h == h]
        if len(recent) >= 2:
            progress = recent[0] - recent[-1]
            if progress < 1e-4 and of > 0.1:
                mu *= 1.02  # not converging and still congested -> stronger penalty

    # Gradient guard: if gradients are exploding, damp growth to keep the solve stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.01)

    new_lambda = current_lambda * mu

    # Final hard clamp to the contractually required range.
    if not (new_lambda == new_lambda):
        new_lambda = current_lambda
    return float(min(max(new_lambda, LO), HI))