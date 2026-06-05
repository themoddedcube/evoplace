def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-penalty schedule.

    Grows lambda multiplicatively (DREAMPlace-style) but modulates the
    growth rate by the overflow trend: push harder while overflow is
    high or stalling, ease off as the layout legalizes so HPWL can
    settle. Hard-clamped to the valid range and guarded against the
    NaN/Inf blow-ups that send normalized_hpwl to inf.
    """
    # Sanitize inputs so a bad upstream value can't poison the schedule.
    if not (current_lambda == current_lambda) or current_lambda in (
        float("inf"),
        float("-inf"),
    ):
        current_lambda = 1.0
    current_lambda = min(max(current_lambda, 0.01), 50.0)

    if not (overflow == overflow) or overflow < 0.0:
        overflow = 1.0

    # Base multiplicative growth with a gentle iteration-based decay,
    # so early iterations ramp the penalty and late ones fine-tune.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow trend: compare recent overflow to a short window back.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        window = min(5, len(overflow_history))
        recent = overflow_history[-1]
        past = overflow_history[-window]
        if past == past and recent == recent:
            trend = past - recent  # >0 means overflow is improving

    # Modulate growth: when overflow is high and not improving, push the
    # penalty up faster; when overflow is low or dropping fast, ease off
    # toward LOWER_PCOF to let wirelength relax.
    if overflow > 0.10:
        if trend <= 1e-4:
            # Stalled at high overflow: accelerate legalization.
            mu = base_mu * 1.03
        else:
            mu = base_mu
    else:
        # Nearly legal: blend toward a near-unity multiplier for HPWL fine-tuning.
        frac = min(max(overflow / 0.10, 0.0), 1.0)
        mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * frac

    # Bound the per-step multiplier to avoid runaway growth.
    mu = min(max(mu, 0.95), 1.10)

    next_lambda = current_lambda * mu

    if not (next_lambda == next_lambda) or next_lambda in (
        float("inf"),
        float("-inf"),
    ):
        next_lambda = current_lambda

    return float(min(max(next_lambda, 0.01), 50.0))