def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty schedule with safe clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric growth (DREAMPlace-style), gently annealed.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push harder when cells are still
    # spread out (high overflow), ease off as the layout legalizes.
    of = overflow if overflow == overflow else 1.0  # NaN guard
    of = min(max(of, 0.0), 1.0)

    if of > 0.5:
        # Far from legal: accelerate density penalty growth.
        mu = base_mu * (1.0 + 0.5 * (of - 0.5))
    elif of > 0.1:
        # Closing in: nominal growth.
        mu = base_mu
    else:
        # Nearly legal: stop inflating, allow slight relaxation so the
        # wirelength gradient can fine-tune placement.
        mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * (of / 0.1)

    # Detect stagnation in overflow trend; if not improving, nudge harder.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] >= recent[0] - 1e-4 and of > 0.1:
            mu *= 1.02

    # Damp explosive growth when gradients blow up.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.01)

    next_lambda = current_lambda * mu

    # Enforce contract: finite float in [0.01, 50.0].
    if next_lambda != next_lambda:
        next_lambda = current_lambda
    return float(min(max(next_lambda, 0.01), 50.0))