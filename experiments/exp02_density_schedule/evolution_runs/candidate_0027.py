def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base geometric growth (DREAMPlace-style), decaying with iteration.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive growth: push density penalty harder while the
    # layout is still congested, ease off as bins clear out.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Map overflow -> growth multiplier in roughly [LOWER_PCOF, UPPER_PCOF*1.06].
    # High overflow -> aggressive ramp; low overflow -> gentle.
    of_mu = LOWER_PCOF + (UPPER_PCOF * 1.06 - LOWER_PCOF) * (of ** 0.5)

    # Detect stalled overflow reduction; if progress stalls while still
    # congested, give an extra nudge to escape the plateau.
    stall_boost = 1.0
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        improvement = recent[0] - recent[-1]
        if improvement < 0.005 and of > 0.10:
            stall_boost = 1.04

    # Blend base schedule with overflow signal (weight base more early,
    # overflow feedback more once it becomes meaningful).
    blend = 0.5
    mu = (1.0 - blend) * base_mu + blend * of_mu
    mu *= stall_boost

    # Once overflow is essentially resolved, stop inflating lambda so the
    # optimizer can fine-tune HPWL at the converged density weight.
    if of < 0.08:
        mu = min(mu, 1.0 + 0.5 * (of / 0.08))

    # Gradient-norm safeguard: if gradients explode, damp the growth to
    # avoid destabilizing the placement.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.01)

    mu = min(max(mu, 0.95), 1.15)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))