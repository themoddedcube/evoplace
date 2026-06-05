def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric growth that decays toward 1.0 as placement matures,
    # mirroring DREAMPlace's density-weight ramp.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push density penalty harder while many
    # bins are over capacity, ease off as overflow collapses so the solver
    # can fine-tune wirelength without over-spreading.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow from history (negative => improving).
    trend = 0.0
    if overflow_history:
        n = min(5, len(overflow_history))
        recent = overflow_history[-n:]
        if len(recent) >= 2 and recent[0] == recent[0]:
            trend = recent[-1] - recent[0]

    if of > 0.10:
        # Still spreading: scale growth with how far overflow exceeds target.
        mu = base_mu * (1.0 + 0.5 * (of - 0.10))
    else:
        # Near target: relax penalty, more so when overflow keeps falling.
        relax = 1.0 if trend >= 0.0 else 1.0 + min(0.5, -trend * 5.0)
        mu = 1.0 + (base_mu - 1.0) * 0.5 / relax

    # Stabilize against exploding/vanishing gradients.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        mu = 1.0 + (mu - 1.0) * 0.5
    elif 0.0 < gn < 1e-3:
        mu = mu * UPPER_PCOF

    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF * 1.5)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:                            # NaN fallback
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))