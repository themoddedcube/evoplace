def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    # Base multiplicative growth, decaying with iteration (DREAMPlace-style).
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive coefficient: push density weight up faster while the
    # placement is still congested, ease off as bins clear out.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Trend from history: if overflow is stalling/rising, accelerate; if it is
    # dropping steadily, relax so wirelength can settle.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-1]
        prev = overflow_history[-min(5, len(overflow_history))]
        trend = recent - prev  # >0 means overflow getting worse

    # Map overflow + trend into [LOWER_PCOF, UPPER_PCOF].
    drive = 0.6 * of + 0.4 * max(min(trend * 10.0 + 0.5, 1.0), 0.0)
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * drive

    mu = pcof * base

    # Gradient-norm safety: if gradients explode, damp the growth to avoid
    # destabilizing the optimizer.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.0 + 0.5 * (mu - 1.0))

    new_lambda = current_lambda * mu

    # Once nearly legal, stop inflating the density penalty so the optimizer can
    # fine-tune wirelength instead of over-spreading.
    if of < 0.08:
        new_lambda = current_lambda * max(min(mu, 1.0), 0.999)

    return float(min(max(new_lambda, 0.01), 50.0))