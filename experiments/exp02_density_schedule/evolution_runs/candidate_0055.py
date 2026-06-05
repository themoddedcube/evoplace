def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LOWER = 0.01
    UPPER = 50.0

    # Sanitize inputs to avoid NaN/inf propagation that blows the run up.
    lam = current_lambda
    if not (lam == lam) or lam in (float("inf"), float("-inf")):
        lam = 1.0
    lam = min(max(lam, LOWER), UPPER)

    ovf = overflow
    if not (ovf == ovf) or ovf in (float("inf"), float("-inf")):
        ovf = 1.0
    ovf = min(max(ovf, 0.0), 1.0)

    # DREAMPlace-style multiplicative density-weight update, but bounded.
    # Base growth rate tapers as iterations progress (smoother late-stage).
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: spread aggressively while bins are congested,
    # then ease off as overflow drops so we can fine-tune wirelength.
    if ovf > 0.10:
        mu = base
    else:
        # Scale growth toward ~1.0 as overflow -> target, so lambda stops
        # ballooning once the layout is legal enough.
        t = ovf / 0.10
        mu = LOWER_PCOF + (base - LOWER_PCOF) * t

    # If overflow is rising vs recent history, push harder; if falling, relax.
    if overflow_history:
        recent = overflow_history[-1]
        if recent == recent and ovf > recent:
            mu *= 1.01
        elif recent == recent and ovf < recent:
            mu *= 0.999

    # Damp updates when gradients explode to keep the optimizer stable.
    gn = gradient_norm
    if gn == gn and gn > 0.0 and gn != float("inf"):
        if gn > 1e3:
            mu = 1.0 + (mu - 1.0) * 0.5

    mu = min(max(mu, 0.5), 1.10)

    new_lambda = lam * mu
    if not (new_lambda == new_lambda):
        new_lambda = lam
    return min(max(new_lambda, LOWER), UPPER)