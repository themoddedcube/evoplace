def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base geometric growth (DREAMPlace-style), annealed with iteration.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push density harder while bins are packed,
    # ease off as the layout spreads so wirelength can be refined.
    of = overflow if overflow == overflow else 1.0      # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Map overflow in [0,1] to a multiplier centered on the base growth.
    # High overflow -> faster ramp; low overflow -> gentle ramp toward 1.
    overflow_factor = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.5)
    mu = 0.5 * base_mu + 0.5 * overflow_factor

    # Trend term: if overflow is stalling (not decreasing), accelerate;
    # if it is dropping fast, decelerate to avoid overshoot.
    if len(overflow_history) >= 3:
        recent = overflow_history[-1]
        prev = overflow_history[-3]
        delta = prev - recent                            # >0 means improving
        if delta < 1e-4:
            mu *= 1.03                                   # stalled: push more
        elif delta > 0.02:
            mu *= 0.985                                  # improving fast: ease

    # Gradient-norm safeguard: if gradients are exploding, dampen the update.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 10.0:
            mu *= 0.97

    # Late-stage fine-tuning: once well spread, stop growing the penalty.
    if of < 0.08:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Clamp to required output range.
    if new_lambda != new_lambda:                         # NaN guard
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))