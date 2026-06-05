def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ ... """
    # Base geometric ramp, gentle and annealed so it can't run away
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive growth: push harder while bins are congested,
    # ease off as the placement spreads out (low overflow -> mu near 1)
    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)

    # Trend from history: if overflow is stalling/rising, grow faster;
    # if it is dropping nicely, slow the penalty so HPWL can settle
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-1]
        if prev == prev:
            trend = of - min(max(prev, 0.0), 1.0)           # >0 = worsening

    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * decay * of
    if trend > 0.0:
        mu *= 1.0 + min(trend, 0.1)                         # react to backsliding
    else:
        mu *= 1.0 + 0.5 * max(trend, -0.05)                 # relax when improving

    # Gradient safety: if gradients blow up, do not amplify them
    if gradient_norm == gradient_norm and gradient_norm > 1e6:
        mu = min(mu, 1.0)

    mu = min(max(mu, 0.95), 1.10)

    base = current_lambda if current_lambda == current_lambda else 1.0
    new_lambda = base * mu

    # Hard clamp to the legal range; also a final NaN/inf backstop
    if not (new_lambda == new_lambda) or new_lambda == float("inf"):
        new_lambda = 50.0
    return float(min(max(new_lambda, 0.01), 50.0))