def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ ... """
    UPPER_PCOF = 1.05

    # Base multiplicative growth that gently decays as iterations progress
    # (same backbone as the incumbent, so we never regress below it).
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Estimate the recent overflow trend to drive an adaptive multiplier.
    if len(overflow_history) >= 3:
        delta = overflow_history[-1] - overflow_history[-3]
    elif len(overflow_history) >= 2:
        delta = overflow_history[-1] - overflow_history[-2]
    else:
        delta = 0.0

    if delta >= 0.0:
        # Overflow stalled or worsening: cells not spreading -> push density
        # penalty harder to escape the plateau.
        mu = base_mu * 1.04
    else:
        # Overflow improving: scale growth down in proportion to progress so we
        # don't over-penalize density once spreading is already underway.
        progress = min(-delta / max(overflow, 1e-6), 1.0)
        mu = base_mu * (1.0 - 0.12 * progress)

    # Late-stage fine-tuning: once placement is nearly legal, stop inflating the
    # density weight so the optimizer can sharpen HPWL on accurate gradients.
    if overflow < 0.10:
        mu = min(mu, 1.0)
    elif overflow < 0.20:
        mu = min(mu, base_mu)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))