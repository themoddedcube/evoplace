def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with bounded multiplicative growth."""
    LOWER, UPPER = 0.01, 50.0

    # Sanitize inputs (guard against NaN/inf propagating into the schedule).
    if not (current_lambda == current_lambda) or current_lambda in (float("inf"), float("-inf")):
        current_lambda = 1.0
    current_lambda = min(max(current_lambda, LOWER), UPPER)

    of = overflow if (overflow == overflow) else 1.0
    of = min(max(of, 0.0), 1.0)

    # Base DREAMPlace-style multiplier: grow the density penalty over time so
    # cells spread out, but decay the growth rate as iterations accumulate.
    UPPER_PCOF, LOWER_PCOF = 1.05, 0.95
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive component: push harder while bins are congested,
    # ease off (and even relax) once the layout is nearly legal so that the
    # late-stage optimization can fine-tune HPWL without over-penalizing.
    # mu interpolates in [LOWER_PCOF, UPPER_PCOF] driven by current overflow.
    target_of = 0.10
    if of > target_of:
        # Congested: scale growth with how far above target we are.
        congestion = min((of - target_of) / (1.0 - target_of), 1.0)
        mu = base_mu * (1.0 + 0.10 * congestion)
    else:
        # Near-legal: blend toward a relaxing multiplier for HPWL fine-tuning.
        relax = 1.0 - (of / target_of)  # 0 at target -> 1 at zero overflow
        mu = base_mu * (1.0 - 0.05 * relax)
        if mu < LOWER_PCOF:
            mu = LOWER_PCOF

    # Trend damping: if overflow is rising despite the penalty, lean on growth;
    # if it is steadily falling, avoid overshooting.
    if len(overflow_history) >= 3:
        recent = [h for h in overflow_history[-3:] if h == h]
        if len(recent) >= 2:
            trend = recent[-1] - recent[0]
            if trend > 0.0:
                mu *= 1.02
            elif trend < -0.02:
                mu *= 0.99

    # Gradient-norm safeguard: if gradients blow up, hold the penalty steady
    # rather than amplifying instability.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e6:
            mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Final clamp into the legal range; never return NaN/inf.
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = current_lambda
    return float(min(max(new_lambda, LOWER), UPPER))