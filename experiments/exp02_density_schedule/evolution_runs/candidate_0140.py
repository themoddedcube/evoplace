def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule (RePlAce-style subgradient update).

    Grows the density penalty geometrically early on, but modulates the growth
    rate by the *trend* of overflow: push harder when spreading stalls, ease off
    once cells are spreading well or overflow is already low so HPWL can relax.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric growth, gently decaying so the earliest iterations push hardest.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    mu = base
    if len(overflow_history) >= 2:
        prev = float(overflow_history[-2])
        cur = float(overflow_history[-1])
        delta = prev - cur  # > 0 means overflow is decreasing (spreading works)
        ref = 0.01
        # Exponent < 1 when overflow drops fast (slow lambda growth);
        # > 1 when overflow stalls or rises (accelerate the penalty).
        exponent = -delta / ref + 1.0
        exponent = min(max(exponent, -1.0), 1.0)
        mu = base ** exponent

    # Near-legal layouts: stop over-penalizing density so wirelength can fine-tune.
    if overflow < 0.10:
        mu = min(mu, 1.0 + 0.5 * (UPPER_PCOF - 1.0))

    # Guard against runaway gradients late in the run.
    if gradient_norm > 0.0 and overflow < 0.05:
        mu = min(mu, 1.0)

    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))