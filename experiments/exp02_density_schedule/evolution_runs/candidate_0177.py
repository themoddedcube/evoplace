def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # DREAMPlace-style multiplicative ramp, decaying step size with iteration
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive growth: push the density penalty harder while many
    # bins are over-dense, ease off as the layout approaches legalization.
    of = min(max(overflow, 0.0), 1.0)
    overflow_factor = 1.0 + 0.5 * of  # 1.0 (legal) .. 1.5 (fully congested)

    # Overflow momentum: react to whether overflow is actually improving.
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        delta = prev - last  # >0 => improving
        if delta <= 1e-4:
            overflow_factor *= 1.15      # stalled -> accelerate penalty growth
        elif delta > 0.02:
            overflow_factor *= 0.90      # improving fast -> decelerate, protect HPWL

    # Gradient safeguard: if gradients are exploding, throttle the increase
    # to keep optimization stable; if very calm, allow a touch more push.
    if gradient_norm > 1e6:
        overflow_factor *= 0.85
    elif gradient_norm < 1.0:
        overflow_factor *= 1.05

    mu = UPPER_PCOF * base * overflow_factor
    mu = min(max(mu, LOWER_PCOF), 1.5)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))