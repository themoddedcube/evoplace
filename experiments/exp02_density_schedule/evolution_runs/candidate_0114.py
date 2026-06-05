def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-weight ramp.

    Keeps DREAMPlace's default improving-HPWL branch as the backbone
    (mu = UPPER_PCOF * max(0.9999^iter, 0.98)) but modulates the step by
    the overflow trend — a hook-visible proxy for the HPWL delta the real
    update uses. While spreading stalls (overflow flat or rising) the
    penalty is pushed harder; once overflow falls steadily the ramp relaxes
    toward 1.0 so low-gamma fine-tuning is not destabilized. Output is
    clamped to keep lambda in a sane range and avoid the divergence that
    sends HPWL to inf.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow trend over the last few hook calls (negative == improving).
    trend = 0.0
    if overflow_history:
        window = overflow_history[-5:]
        if len(window) >= 2:
            trend = overflow - window[0]
        ref = max(abs(window[0]), 1e-3)
    else:
        ref = max(overflow, 1e-3)

    # Push harder when stalled/rising, relax when overflow is dropping fast.
    adapt = UPPER_PCOF ** (max(min(trend / ref, 1.0), -1.0))
    mu = base_mu * adapt

    # Ease off the penalty in the low-overflow fine-tuning regime.
    if overflow < 0.1:
        mu = 1.0 + (mu - 1.0) * (overflow / 0.1)

    # Damp explosive growth when gradients are already large.
    if gradient_norm > 0.0 and mu > 1.0:
        mu = 1.0 + (mu - 1.0) / (1.0 + 0.01 * gradient_norm)

    mu = max(min(mu, UPPER_PCOF * UPPER_PCOF), LOWER_PCOF)

    new_lambda = current_lambda * mu
    return max(0.01, min(50.0, new_lambda))