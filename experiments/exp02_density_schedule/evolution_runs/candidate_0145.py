def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-weight schedule.

    Grows lambda fast while cells are still spread out (high overflow),
    then tapers the growth as the layout legalizes so HPWL can settle.
    The per-step multiplier is modulated by the recent overflow trend:
    if overflow stalls or rises, push harder; if it is falling nicely,
    ease off to avoid overshooting and disturbing wirelength.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.003

    # Base decaying ceiling on the multiplier (DREAMPlace-style).
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-driven aggressiveness: high overflow -> larger multiplier.
    # Map overflow in [0, 1] onto [LOWER_PCOF, UPPER_PCOF].
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = 0.0 if of < 0.0 else (1.0 if of > 1.0 else of)
    overflow_factor = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Trend term from recent history: if overflow is not improving,
    # nudge the multiplier up; if it is dropping, relax it.
    trend = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        recent = overflow_history[-min(5, len(overflow_history)):]
        delta = recent[0] - recent[-1]  # positive => improving
        span = len(recent) - 1
        if span > 0:
            rate = delta / span
            # rate>0 (improving) lowers mu slightly; rate<0 raises it.
            trend = -3.0 * rate

    # Gradient sanity: if gradients explode, be gentler to stay stable.
    grad_damp = 1.0
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1.0:
            grad_damp = 1.0 / (1.0 + 0.05 * (gradient_norm - 1.0))

    mu = base * overflow_factor * (1.0 + trend) * grad_damp

    # Keep the per-step change bounded for numerical stability.
    if mu < 0.97:
        mu = 0.97
    elif mu > UPPER_PCOF:
        mu = UPPER_PCOF

    new_lambda = current_lambda * mu

    # Enforce the required output range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return new_lambda