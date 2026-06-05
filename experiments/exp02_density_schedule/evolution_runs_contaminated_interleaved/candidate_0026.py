def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with bounded, stall-aware growth."""
    # --- sanitize inputs ---
    cl = current_lambda if (current_lambda is not None and current_lambda == current_lambda) else 1.0
    cl = min(max(cl, 0.01), 50.0)
    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(max(ov, 0.0), 1.0)
    gn = gradient_norm if (gradient_norm is not None and gradient_norm == gradient_norm) else 1.0

    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base decay term (mirrors DREAMPlace), kept strictly bounded.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive multiplier: push density penalty hard while spread is
    # poor (high overflow), ease off as the layout legalizes (low overflow).
    # Maps overflow in [0,1] -> growth factor in [LOWER_PCOF, UPPER_PCOF].
    target = 0.10  # desired terminal overflow
    drive = (ov - target) / (1.0 - target)
    drive = min(max(drive, 0.0), 1.0)
    mu = LOWER_PCOF + (base - LOWER_PCOF) * drive

    # Stall detection: if overflow has plateaued, nudge growth up to escape.
    if overflow_history and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if all(r == r for r in recent):
            improvement = recent[0] - recent[-1]
            if improvement < 1e-4 and ov > target:
                mu *= 1.02

    # Damp growth when gradients explode to keep optimization stable.
    if gn != gn or gn > 1e6:
        mu = min(mu, 1.0)

    next_lambda = cl * mu

    # Hard clamp to the legal range.
    if next_lambda != next_lambda:
        next_lambda = cl
    return min(max(next_lambda, 0.01), 50.0)