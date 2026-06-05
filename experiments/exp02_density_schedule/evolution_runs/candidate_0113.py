def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight growth with safe clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    # NaN/inf-safe overflow in [0, 1]
    of = overflow
    if of != of or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    # Geometric base that decays toward a floor as the run progresses,
    # so early iterations spread cells aggressively, later ones gently.
    base_mu = max(0.9999 ** float(iteration), 0.98)

    # Scale growth by how congested the layout still is: many overfull
    # bins -> push harder; nearly legal -> grow slowly.
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of
    mu = pcof * base_mu

    # Trend term: if overflow is rising, accelerate; if falling, relax.
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev and prev not in (float("inf"), float("-inf")):
            delta = of - min(max(prev, 0.0), 1.0)
            mu *= 1.0 + 0.5 * max(min(delta, 0.1), -0.05)

    # Gradient guard: if gradients explode, damp the increase.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.0 + 0.5 * (mu - 1.0))

    new_lambda = current_lambda * mu

    # Once the placement is essentially legal, freeze lambda and let the
    # optimizer refine HPWL instead of over-penalizing density.
    if of < 0.06:
        new_lambda = current_lambda

    # Final safety: replace NaN and clamp to the valid range.
    if new_lambda != new_lambda:
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))