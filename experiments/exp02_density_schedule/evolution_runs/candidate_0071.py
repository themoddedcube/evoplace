def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplicative ramp (DREAMPlace-style), decaying with iteration.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push density penalty harder while many bins are
    # over-filled, ease off as the layout legalizes so HPWL can be refined.
    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)

    # Overflow trend from recent history: if overflow is stalling/rising,
    # accelerate; if it is dropping fast, relax to avoid overshoot.
    trend = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last:
            trend = last - prev   # >0 means overflow growing (bad)

    # Map overflow magnitude to a multiplier in roughly [LOWER, UPPER].
    # High overflow -> stronger growth; low overflow -> gentle decay.
    of_factor = 0.5 + of                                    # in [0.5, 1.5]
    if trend > 1e-4:
        of_factor *= 1.05                                   # stalled: push
    elif trend < -1e-3:
        of_factor *= 0.97                                   # converging: ease

    mu = base_mu * of_factor

    # Gradient-norm safeguard: if gradients explode, damp the step to keep
    # the optimization stable; if they are tiny, allow a touch more push.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu *= 0.9
        elif gradient_norm < 1e-3:
            mu *= 1.02

    # Keep per-step change bounded for stability.
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF * 1.5)

    new_lambda = current_lambda * mu

    # Clamp to the required output range.
    if new_lambda != new_lambda:                            # NaN guard
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))