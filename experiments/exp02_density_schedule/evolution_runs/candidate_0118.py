def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative schedule with stall detection."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplier: decays the cap from ~1.05 toward ~0.98 over iterations
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling. High overflow -> cells still spread out, so
    # push lambda harder (faster density penalty growth). Low overflow ->
    # nearly legal, ease off so HPWL can be fine-tuned without overshoot.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Map overflow in [0,1] to a multiplier shaping factor in [LOWER, UPPER].
    shape = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Stall detection: if overflow has stopped improving, nudge lambda up to
    # break out of the plateau; if it's dropping fast, hold steady.
    boost = 1.0
    if overflow_history is not None and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[0] - recent[-1]  # positive => improving
        if delta < 1e-4:
            boost = 1.02  # stalled, accelerate
        elif delta > 0.02:
            boost = 0.99  # converging well, gentle

    # Gradient-norm guard: if gradients explode, damp the update to stay stable.
    damp = 1.0
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        damp = 0.97

    mu = base * shape * boost * damp
    new_lambda = current_lambda * mu

    # Clamp to required output range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)