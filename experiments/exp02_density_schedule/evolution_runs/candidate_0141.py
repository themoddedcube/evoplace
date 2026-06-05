def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative lambda schedule with divergence guards."""
    # Sanitize inputs
    ov = overflow if overflow == overflow else 1.0          # NaN guard
    ov = min(max(ov, 0.0), 1.0)
    cur = current_lambda if current_lambda == current_lambda else 1.0
    cur = min(max(cur, 0.01), 50.0)

    # Base DREAMPlace-style decaying multiplier: aggressive early, gentle late.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: high overflow -> push density harder,
    # low overflow (cells spread) -> ease off so HPWL can be fine-tuned.
    # Map overflow in [0,1] to a multiplier exponent.
    spread_factor = min(max((ov - 0.10) / 0.90, 0.0), 1.0)
    mu = LOWER_PCOF + (base - LOWER_PCOF) * spread_factor

    # Trend awareness: if overflow is stalling (not decreasing), nudge harder.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        prev_avg = sum(recent[:-1]) / max(len(recent) - 1, 1)
        if ov >= prev_avg - 1e-4 and ov > 0.10:
            mu *= 1.01  # break the plateau
        elif ov < 0.05:
            mu = min(mu, 1.002)  # nearly converged: hold steady

    # Gradient-norm safety: if gradients explode, do not amplify lambda.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.0)

    new_lambda = cur * mu

    # Hard clamp to required output range.
    if new_lambda != new_lambda:
        new_lambda = cur
    return float(min(max(new_lambda, 0.01), 50.0))