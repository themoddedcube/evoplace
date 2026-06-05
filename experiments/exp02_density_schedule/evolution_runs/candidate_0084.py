def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ ... """
    # Overflow-adaptive multiplicative density-weight update (DREAMPlace-style),
    # with trend sensing and gradient-norm stabilization.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplier: decays toward 1 as iterations progress so growth
    # slows down in the fine-tuning phase (mirrors the original schedule).
    base_decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive component. When overflow is high, cells are still
    # heavily clustered -> push the penalty up faster. When overflow is low,
    # placement is nearly legal -> ease off so HPWL can refine.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow from history: if overflow is stalling (not dropping),
    # increase pressure; if it is dropping fast, relax to avoid overshoot.
    trend = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        recent = overflow_history[-1]
        prev = overflow_history[min(len(overflow_history), 5)]  # ~5 steps back if available
        # fall back to the earliest available sample
        prev = overflow_history[-min(len(overflow_history), 5)]
        delta = prev - recent  # positive => overflow decreasing (good progress)
        trend = delta

    # Map overflow magnitude to a growth factor in [LOWER_PCOF, UPPER_PCOF].
    # high overflow -> closer to UPPER_PCOF, low overflow -> closer to 1.
    mu = 1.0 + (UPPER_PCOF - 1.0) * (of ** 0.5)

    # Stall correction: if overflow is not decreasing, nudge mu upward.
    if trend <= 1e-4:
        mu *= 1.02
    elif trend > 0.02:
        # progressing quickly; relax so we don't over-penalize density
        mu *= 0.99

    # Gradient-norm stabilization: if gradients explode, damp the growth to
    # keep the optimization stable; if they vanish, allow a touch more push.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 0.0:
        if gn > 10.0:
            mu = 1.0 + (mu - 1.0) * 0.5
        elif gn < 0.1:
            mu *= 1.01

    mu *= base_decay

    # Clamp the per-step multiplier into a safe band.
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    new_lambda = current_lambda * mu

    # Final hard clamp on the returned value.
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))