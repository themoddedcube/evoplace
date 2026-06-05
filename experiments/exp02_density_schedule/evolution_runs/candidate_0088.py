def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """Overflow-adaptive multiplicative density-weight schedule with safety clamps."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Geometric backbone: aggressive early growth, decaying toward a floor.
    # Pushes cells apart (spreads density) quickly, then eases off for fine-tuning.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow trend: react to how density is actually evolving.
    mu = base_mu
    if overflow_history:
        prev = overflow_history[-1]
        delta = overflow - prev  # negative => overflow shrinking (good)
        if delta < 0.0:
            # Density improving: ramp lambda a touch faster to keep momentum.
            mu = base_mu * (1.0 + min(0.10, (-delta) * 3.0))
        else:
            # Density stuck or worsening: throttle growth to avoid overshoot/divergence.
            mu = base_mu * max(LOWER_PCOF, 1.0 - delta * 3.0)

    # Late-stage relaxation: once nearly legal, stop inflating lambda so HPWL
    # can settle instead of being dominated by the density penalty.
    if overflow < 0.10:
        mu = min(mu, 1.0 + overflow)

    # Gradient safety: if the field is exploding, do not amplify it further.
    if gradient_norm and gradient_norm > 1.0e3:
        mu = min(mu, 1.0)

    # Numerical guards.
    if not (mu == mu) or mu <= 0.0:   # NaN / non-positive guard
        mu = 1.0

    new_lambda = current_lambda * mu
    if not (new_lambda == new_lambda):  # NaN guard
        new_lambda = current_lambda

    return float(min(max(new_lambda, 0.01), 50.0))