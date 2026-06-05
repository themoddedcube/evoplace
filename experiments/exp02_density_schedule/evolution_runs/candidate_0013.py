def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base geometric ramp (DREAMPlace-style), gentler as iterations grow.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: push harder when cells are still spread (high overflow),
    # ease off as overflow collapses so we don't over-penalize near convergence.
    ov = overflow if overflow is not None else 1.0
    ov = min(max(ov, 0.0), 1.0)

    # Trend from history: if overflow is dropping fast, relax the ramp.
    delta = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        delta = overflow_history[-2] - overflow_history[-1]  # >0 means improving

    if ov > 0.10:
        # Still legalizing: scale ramp up with remaining overflow.
        mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * (0.5 + 0.5 * ov)
        if delta > 0.02:
            mu = 1.0 + (mu - 1.0) * 0.7  # improving fast, don't overshoot
    else:
        # Near-legal: fine-tune, keep lambda nearly flat to refine HPWL.
        mu = 1.0 + (base_mu - 1.0) * (ov / 0.10) * 0.5
        mu = max(mu, LOWER_PCOF)

    # Gradient guard: if gradients blow up, damp the multiplier.
    if gradient_norm is not None and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = current_lambda * mu

    # Hard clamp to the required range; guard against NaN/inf.
    if not (new_lambda == new_lambda) or new_lambda == float("inf"):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))