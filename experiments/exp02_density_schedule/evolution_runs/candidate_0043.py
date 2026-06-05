def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with hard bounds."""
    LOWER, UPPER = 0.01, 50.0

    # Sanitize inputs (NaN/inf guards) so we never propagate inf.
    lam = current_lambda
    if not (lam == lam) or lam in (float("inf"), float("-inf")):
        lam = 1.0
    lam = min(max(lam, LOWER), UPPER)

    ovf = overflow
    if not (ovf == ovf) or ovf < 0.0:
        ovf = 1.0
    ovf = min(ovf, 1.0)

    # Base multiplicative growth, decaying with iteration (ePlace/RePlAce style).
    base = max(0.9999 ** float(iteration), 0.98)

    # Adapt growth to overflow trend: push harder while spreading is poor,
    # ease off (and gently shrink) once overflow is low and still falling.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev:  # not NaN
            trend = ovf - prev  # >0 worsening, <0 improving

    if ovf > 0.10:
        # High overflow: grow weight, more aggressively if not improving.
        mu = 1.05 * base
        if trend > 0.0:
            mu *= 1.02
    else:
        # Near-converged: stop ramping, settle toward fine-tuning.
        mu = 1.0 * base if trend < 0.0 else 1.01 * base

    # Dampen if gradients explode to keep optimization stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.0)

    lam = lam * mu
    return float(min(max(lam, LOWER), UPPER))