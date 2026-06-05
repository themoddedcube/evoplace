def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Density-penalty (lambda) schedule for DREAMPlace.
    # Grows lambda to push cells apart while overflow is high, then
    # eases off and holds steady as the layout legalizes. All paths
    # are clamped to the required [0.01, 50.0] range to avoid blow-up.

    LO, HI = 0.01, 50.0

    # Base multiplicative growth, gentler than the stock 1.05 so the
    # penalty ramps without overshooting into the inf regime.
    base_mu = 1.04

    # Overflow-adaptive gain: when many bins are overfull we still need
    # to spread cells, so push harder; when overflow is low, back off
    # toward 1.0 (hold) so wirelength can be fine-tuned.
    of = overflow if overflow == overflow else 1.0  # NaN guard
    of = min(max(of, 0.0), 1.0)
    adapt = 0.5 + 1.0 * of  # in [0.5, 1.5]

    # Trend term: if overflow stopped improving, nudge the penalty up a
    # touch to break the stall; if it is dropping fast, relax.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-1]
        if prev == prev:
            delta = of - min(max(prev, 0.0), 1.0)
            if delta > -1e-4:        # plateau or worsening
                trend = 0.02
            elif delta < -0.02:      # improving quickly
                trend = -0.02

    mu = 1.0 + (base_mu - 1.0) * adapt + trend

    # Gradient safety: if gradients are exploding, damp the growth.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e6:
        mu = min(mu, 1.01)

    # Once the layout is nearly legal, stop growing and gently decay so
    # the density force does not dominate final HPWL refinement.
    if of < 0.10:
        mu = 0.98

    cl = current_lambda if current_lambda == current_lambda else 1.0
    cl = min(max(cl, LO), HI)

    next_lambda = cl * mu
    return float(min(max(next_lambda, LO), HI))