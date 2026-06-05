def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base DREAMPlace-style geometric growth, decaying with iteration.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: when many bins are over-dense we still
    # need spreading force, so grow faster; once overflow is low we ease off
    # to let HPWL refine without over-penalizing density.
    ov = overflow if overflow == overflow else 1.0  # guard NaN
    ov = min(max(ov, 0.0), 1.0)

    # Trend of overflow from history (negative => improving / spreading well).
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        cur = overflow_history[-1]
        if prev == prev and cur == cur:
            trend = cur - prev

    # Target multiplier interpolates between mild shrink and the base growth
    # according to how far overflow is above a healthy threshold (~0.10).
    spread_need = min(max((ov - 0.10) / 0.90, 0.0), 1.0)
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * spread_need

    # If overflow is rising (cells re-clumping) push a bit harder.
    if trend > 0.0:
        mu *= 1.0 + min(trend, 0.5)
    elif trend < 0.0:
        mu *= 1.0 - 0.5 * min(-trend, 0.1)

    # Gradient-norm safety: if gradients explode, damp lambda growth to keep
    # the optimization stable instead of diverging to inf.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1.0e3:
        mu = min(mu, 1.0)

    # Keep the per-step multiplier in a sane band to avoid runaway/collapse.
    mu = min(max(mu, 0.90), UPPER_PCOF * 1.2)

    cl = current_lambda if current_lambda == current_lambda else 1.0
    new_lambda = cl * mu

    # Clamp to the allowed range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)