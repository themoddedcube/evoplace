def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-penalty schedule."""
    LOWER, UPPER = 0.01, 50.0

    # Sanitize current lambda (guard against inf/nan / out-of-range state).
    lam = current_lambda
    if not (lam == lam) or lam in (float("inf"), float("-inf")):
        lam = 1.0
    lam = min(max(lam, LOWER), UPPER)

    # Sanitize overflow into [0, 1].
    ov = overflow
    if not (ov == ov):
        ov = 1.0
    ov = min(max(ov, 0.0), 1.0)

    # Base geometric growth (DREAMPlace-style), gently decaying with iteration
    # so the penalty ramps fast early and eases as the layout settles.
    UPPER_PCOF = 1.05
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive boost: when many bins are over-dense, push the penalty
    # harder; once the design is nearly legal (low overflow), back off so HPWL
    # can be fine-tuned instead of over-spreading.
    boost = 1.0 + 0.25 * (ov - 0.10)

    # Stagnation detection: if overflow has stopped dropping over the recent
    # window, accelerate to escape the plateau.
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        if recent[0] - recent[-1] < 1e-4:
            boost *= 1.05

    # Near-convergence damping: hold lambda steady once overflow is very small
    # to avoid blowing past the optimal trade-off point.
    if ov < 0.06:
        boost = min(boost, 1.005)

    mu = base * boost
    # Keep the per-step multiplier sane to avoid runaway growth / divergence.
    mu = min(max(mu, 0.95), 1.20)

    new_lambda = lam * mu
    return min(max(new_lambda, LOWER), UPPER)