def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Overflow-adaptive density-weight ramp with hard safety clamps.
    # Goal: grow lambda fast while cells are badly spread (high overflow),
    # ease off as the layout legalizes (low overflow), and never blow up.

    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Base multiplier: DREAMPlace-style, but bounded and decaying with iter.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow term: push harder when overflow is high, gentle near convergence.
    # Maps overflow in [0,1] -> growth factor in [LOWER_PCOF, UPPER_PCOF].
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.5) * decay

    # Trend awareness: if overflow has stalled (not dropping), nudge harder.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] >= recent[0] - 1e-4:  # no progress in last few iters
            mu *= 1.02

    # Gradient safety: if gradients are exploding, do not amplify lambda.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.0)

    cl = current_lambda if current_lambda == current_lambda else 1.0
    if cl <= 0.0:
        cl = 0.01

    new_lambda = cl * mu

    # Hard clamp to the required output range.
    if new_lambda != new_lambda:  # NaN
        new_lambda = cl
    return float(min(max(new_lambda, 0.01), 50.0))