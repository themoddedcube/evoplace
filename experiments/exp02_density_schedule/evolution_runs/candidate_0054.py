def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base geometric growth that decays as the placement matures.
    # Early iterations push the density penalty up aggressively; the
    # floor (0.98) keeps a gentle upward pressure late in the run.
    mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive correction: react to the *trend* of overflow.
    # If overflow is still rising (cells not yet spreading), grow lambda
    # faster; if it is falling nicely, ease off so the optimizer can
    # refine wirelength instead of over-penalizing density.
    if overflow_history is not None and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        cur = overflow_history[-1]
        delta = cur - prev
        if delta > 0.0:
            mu *= 1.0 + min(delta * 3.0, 0.10)      # accelerate, capped
        elif delta < 0.0:
            mu *= max(1.0 + delta * 1.5, 0.97)      # decelerate, floored
    elif overflow > 0.0:
        # No history yet: scale push by absolute overflow level.
        mu *= 1.0 + min(overflow * 0.05, 0.05)

    # Stability guard: if gradients are blowing up, stop growing the
    # penalty (and slightly back off) to avoid divergence.
    if gradient_norm is not None and gradient_norm > 0.0:
        if gradient_norm > 1.0e3:
            mu = min(mu, 0.99)
        elif gradient_norm > 1.0e2:
            mu = min(mu, 1.0)

    # Once placement is essentially legal, hold steady and let it settle.
    if overflow <= 0.08:
        mu = min(mu, LOWER_PCOF)

    new_lambda = current_lambda * mu
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)