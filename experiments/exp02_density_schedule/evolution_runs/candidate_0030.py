def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """ ... """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base time-decayed growth: aggressive early, gentler as placement matures.
    mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: push the density penalty harder while bins are
    # congested, ease off once the layout is nearly legal so HPWL can settle.
    if overflow > 0.9:
        mu *= 1.08
    elif overflow > 0.5:
        mu *= 1.02
    elif overflow < 0.1:
        mu *= LOWER_PCOF

    # Stagnation detection: if overflow has barely moved over the recent
    # window, ramp the penalty to break the plateau.
    if len(overflow_history) >= 5:
        recent = overflow_history[-5:]
        delta = recent[0] - recent[-1]
        if delta < 0.005:
            mu *= 1.05
        elif delta < 0.0:           # overflow getting worse -> push harder
            mu *= 1.07

    # Gradient guard: damp the multiplier when gradients are large to avoid
    # divergence (the cause of inf-valued runs).
    if gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.01)
        elif gradient_norm > 1e3:
            mu = min(mu, 1.03)

    # Keep the per-step update bounded for numerical stability.
    mu = min(max(mu, 0.90), 1.20)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))