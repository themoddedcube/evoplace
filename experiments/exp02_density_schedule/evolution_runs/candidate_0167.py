def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Robust, overflow-adaptive multiplicative density-weight schedule.
    # Grows lambda to drive cells apart early, then eases off as the
    # layout legalizes (overflow -> 0), with hard clamping for stability.

    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.003

    # Sanitize inputs (guard against NaN/inf that caused divergence).
    if not (current_lambda == current_lambda) or current_lambda in (
        float("inf"),
        float("-inf"),
    ):
        current_lambda = 1.0
    current_lambda = min(max(current_lambda, 0.01), 50.0)

    ov = overflow if (overflow == overflow) else 1.0
    ov = min(max(ov, 0.0), 1.0)

    # Base decay of the growth factor over time (DREAMPlace-style).
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive interpolation: high overflow -> aggressive growth
    # (UPPER_PCOF) to spread cells; low overflow -> near-unity (LOWER_PCOF)
    # so the weight stops climbing once density is satisfied.
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * ov

    # Trend damping: if overflow is already falling, don't over-push.
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev and ov < prev:
            pcof = 1.0 + (pcof - 1.0) * 0.6

    # Gradient safety: if gradients blow up, hold the weight steady.
    gn = gradient_norm if (gradient_norm == gradient_norm) else 0.0
    if gn > 1e6:
        pcof = min(pcof, 1.0)

    mu = pcof * base

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))