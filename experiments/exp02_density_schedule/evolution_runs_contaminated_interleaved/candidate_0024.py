def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with divergence guards."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # --- Sanitize inputs (the inf run came from an unguarded update) ---
    if (current_lambda is None) or (current_lambda != current_lambda) or (current_lambda <= 0.0):
        current_lambda = 1.0
    of = overflow if (overflow == overflow) else 1.0
    if of < 0.0:
        of = 0.0
    elif of > 1.0:
        of = 1.0

    # --- Base multiplicative growth with slow decay (DREAMPlace-style) ---
    base = max(0.9999 ** float(iteration), 0.98)

    # --- Overflow-adaptive growth: push hard while cells are clustered
    #     (high overflow), ease off as they spread so lambda doesn't blow up.
    growth = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # --- Trend term: react to the overflow trajectory, not just its level ---
    if overflow_history and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        trend = recent[-1] - recent[0]  # negative => improving
        if trend < 0.0:
            # steady improvement: damp growth toward 1.0 to avoid overshoot
            growth = 1.0 + (growth - 1.0) * 0.5
        elif trend > 0.01:
            # stalling / worsening spread: push a little harder
            growth = growth * 1.02

    # --- Gradient guard: if gradients explode, shrink the update for stability ---
    if (gradient_norm == gradient_norm) and (gradient_norm > 1.0e3):
        growth = 1.0 + (growth - 1.0) * 0.25

    mu = growth * base
    new_lambda = current_lambda * mu

    # --- NaN / range safety ---
    if new_lambda != new_lambda:
        new_lambda = current_lambda
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)