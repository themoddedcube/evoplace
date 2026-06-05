def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive, bounded multiplicative lambda schedule."""
    LO, HI = 0.01, 50.0
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Sanitize inputs (guard against NaN/inf propagating to an out-of-range return)
    cl = current_lambda
    if not (cl == cl) or cl in (float("inf"), float("-inf")):
        cl = 1.0
    of = overflow
    if not (of == of) or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    # Base growth: aggressive early, decaying toward a gentle floor as iters increase.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow gating: push hard while many bins are over-dense, ease off as cells
    # spread out so we don't over-penalize during fine-tuning.
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * of

    # Stall detection: if overflow stops improving while still high, increase pressure.
    if len(overflow_history) >= 3 and of > 0.1:
        recent = overflow_history[-1]
        prev = overflow_history[-3]
        if recent >= prev - 1e-6:
            mu *= 1.03

    # Gradient safety: if gradients explode, damp the update to stay stable.
    gn = gradient_norm
    if gn == gn and gn not in (float("inf"), float("-inf")) and gn > 1e4:
        mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = cl * mu
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cl

    return min(max(new_lambda, LO), HI)