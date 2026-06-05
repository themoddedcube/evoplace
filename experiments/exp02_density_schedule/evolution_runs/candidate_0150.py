def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LOWER = 0.01
    UPPER = 50.0

    # Sanitize incoming state: a non-finite or non-positive lambda would
    # otherwise propagate inf/nan forever (the cause of the inf score).
    lam = current_lambda
    if not (lam == lam) or lam in (float("inf"), float("-inf")) or lam <= 0.0:
        lam = 1.0
    lam = min(max(lam, LOWER), UPPER)

    of = overflow
    if not (of == of) or of < 0.0:
        of = 1.0
    of = min(of, 1.0)

    # Base geometric ramp (DREAMPlace-style), gently annealed over iterations
    # so early steps push density correction hard and late steps fine-tune.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    decay = max(0.9999 ** float(iteration), 0.98)
    base_mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * decay

    # Overflow-adaptive boost: spread cells faster while bins are congested,
    # back off toward 1.0 (hold) as the placement legalizes.
    mu = base_mu * (1.0 + 0.30 * of)

    # Stall detection: if overflow has flattened out, nudge lambda harder to
    # break out of the plateau.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if max(recent) - min(recent) < 1e-3 and of > 0.10:
            mu *= 1.10

    # Gradient guard: if gradients are exploding, slow the ramp for stability.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = 1.0 + (mu - 1.0) * 0.5

    nxt = lam * mu

    if not (nxt == nxt) or nxt in (float("inf"), float("-inf")):
        nxt = lam
    return float(min(max(nxt, LOWER), UPPER))