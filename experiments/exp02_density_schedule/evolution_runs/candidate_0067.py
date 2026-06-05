def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with stagnation handling."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base DREAMPlace-style multiplier: aggressive early, gentle late.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push harder while bins are congested,
    # ease off once cells have spread so HPWL can be refined.
    ovf = overflow if overflow == overflow else 1.0   # guard against NaN
    ovf = min(max(ovf, 0.0), 1.0)
    # Map overflow in [0,1] -> multiplier in [LOWER_PCOF, base].
    mu = LOWER_PCOF + (base - LOWER_PCOF) * (ovf ** 0.5)

    # Stagnation / oscillation detection from recent overflow trend.
    if len(overflow_history) >= 5:
        recent = overflow_history[-5:]
        delta = recent[0] - recent[-1]          # positive => improving
        if delta < 1e-4 and recent[-1] > 0.1:
            mu *= 1.05                            # stuck & congested: push
        elif delta < 0.0:
            mu *= 0.97                            # overflow rising: back off

    # Gradient safeguard: if gradients explode, soften the update.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.02)

    new_lambda = current_lambda * mu

    # Guard against NaN/inf and clamp to legal range.
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))