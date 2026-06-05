def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative penalty schedule with hard clamping.

    Grows the density penalty when overflow is high (cells still spread out)
    and eases growth as overflow drops, so the penalty does not run away to inf.
    """
    LO, HI = 0.01, 50.0

    # Sanitize inputs (NaN/inf-safe).
    lam = current_lambda
    if not (lam == lam) or lam in (float("inf"), float("-inf")):
        lam = 1.0
    lam = min(max(lam, LO), HI)

    ovf = overflow
    if not (ovf == ovf) or ovf < 0.0:
        ovf = 1.0
    ovf = min(ovf, 1.0)

    # Overflow-adaptive growth: aggressive while spread out, gentle near target.
    # Maps overflow in [0,1] -> multiplier in roughly [1.00, 1.08].
    base = 1.0 + 0.08 * (ovf ** 1.5)

    # Stall detection: if overflow stopped improving, push harder.
    if overflow_history is not None and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-3 and ovf > 0.1:
            base *= 1.03

    # Anneal the growth rate down over iterations for late-stage fine-tuning.
    decay = max(0.9995 ** float(iteration), 0.97)
    mu = 1.0 + (base - 1.0) * decay

    # Damp growth when gradients explode to avoid oscillation.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = 1.0 + (mu - 1.0) * 0.5

    new_lam = lam * mu
    if not (new_lam == new_lam):
        new_lam = lam
    return float(min(max(new_lam, LO), HI))