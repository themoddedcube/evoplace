def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty schedule.

    Grows lambda aggressively while the layout is congested (high overflow)
    and tapers the growth as overflow drains, so wirelength fine-tunes once
    cells have spread. Falls back to a gentle ramp on degenerate inputs.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # Sanitize inputs (NaN/inf/negatives) so we always return a valid float.
    if not (current_lambda == current_lambda) or current_lambda <= 0.0:
        current_lambda = 0.01
    of = overflow
    if not (of == of) or of < 0.0:
        of = 1.0
    of = min(of, 1.0)

    # Base time-decay of the growth factor (same spirit as the original).
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive multiplier: push hard while congested, ease off as
    # bins clear. Maps overflow in [0,1] -> coef in [LOWER_PCOF, UPPER_PCOF].
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Trend term: if overflow is still rising vs. recent history, nudge growth
    # up; if it is falling steadily, relax to let HPWL settle.
    if overflow_history and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        valid = [h for h in recent if h == h and h >= 0.0]
        if len(valid) >= 2:
            trend = valid[-1] - valid[0]
            coef *= 1.0 + 0.25 * max(min(trend, 0.4), -0.4)

    # Gradient guard: if gradients explode, dampen the step to stay stable.
    gn = gradient_norm
    if gn == gn and gn > 0.0:
        coef *= 1.0 / (1.0 + 0.05 * max(gn - 1.0, 0.0))

    mu = coef * base

    # Once nearly converged (low overflow), stop inflating lambda so the
    # optimizer can minimize true HPWL.
    if of < 0.10:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    if not (new_lambda == new_lambda):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))