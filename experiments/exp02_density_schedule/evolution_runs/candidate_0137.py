def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Density-weight (lambda) multiplier schedule for DREAMPlace-style global
    # placement. We grow lambda over iterations to drive overflow down, but make
    # the growth overflow-adaptive and gradient-aware so we neither stall (too
    # slow -> never legalizes) nor explode (too fast -> wirelength blows up).

    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base multiplicative growth, gently annealed with iteration (DREAMPlace base).
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push harder while many bins are overfull, ease off
    # as the layout spreads out. overflow in [0, 1]; ~0.1 is a typical target.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)
    target_of = 0.10

    if of > target_of:
        # Far from target -> accelerate, scaled by how far over we are.
        excess = (of - target_of) / (1.0 - target_of + 1e-12)
        mu = base_mu * (1.0 + 0.5 * excess)
    else:
        # Near/under target -> relax growth toward 1.0 so we stop perturbing
        # a good layout and let wirelength settle.
        ratio = of / (target_of + 1e-12)
        mu = 1.0 + (base_mu - 1.0) * ratio

    # Trend awareness: if overflow stopped improving, nudge growth up a touch
    # to break the plateau; if it's dropping fast, don't over-push.
    if len(overflow_history) >= 3:
        recent = overflow_history[-1]
        prev = overflow_history[-3]
        if recent == recent and prev == prev:
            delta = prev - recent  # positive = improving
            if delta < 1e-4 and of > target_of:
                mu *= 1.02
            elif delta > 0.02:
                mu *= 0.99

    # Gradient-norm guard: if gradients are exploding, damp the multiplier to
    # keep the optimization stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.0 + (mu - 1.0) * 0.5)

    # Keep each step bounded and never shrink below the DREAMPlace floor.
    mu = min(max(mu, LOWER_PCOF), 1.10)

    new_lambda = current_lambda * mu

    # Hard clamp to the required output range.
    if new_lambda != new_lambda:  # NaN safety
        new_lambda = 1.0
    return float(min(max(new_lambda, 0.01), 50.0))