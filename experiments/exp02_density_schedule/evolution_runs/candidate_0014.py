def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # NaN/range guards on inputs
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0

    # Base multiplicative growth that anneals toward ~1.0 as iterations proceed.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow trend over recent history (negative => cells are spreading well).
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        k = min(len(overflow_history), 4)
        recent = overflow_history[-1]
        past = overflow_history[-k]
        if recent == recent and past == past:
            trend = recent - past

    # Push lambda harder while bins stay congested and overflow stalls;
    # ease the growth as the layout legalizes (overflow -> 0).
    stall_boost = 1.02 if (trend >= -1e-3 and of > 0.10) else 1.0

    # Blend toward UPPER_PCOF under high overflow, gentle growth under low overflow.
    mu = (LOWER_PCOF + (base_mu - LOWER_PCOF) * (0.5 + 0.5 * of)) * stall_boost

    # Dampen the update if gradients explode (prevents lambda blow-up -> inf HPWL).
    if gn > 1e3:
        mu = min(mu, 1.01)
    mu = min(max(mu, 0.90), UPPER_PCOF * 1.05)

    new_lambda = current_lambda * mu

    # Final NaN/inf guard + clamp to the allowed range.
    if not (new_lambda == new_lambda):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))