def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ Overflow-adaptive multiplicative density-weight schedule. """
    UPPER_PCOF = 1.08
    LOWER_BASE = 0.98

    # DREAMPlace-style decaying envelope on the growth rate
    base = max(0.9999 ** float(iteration), LOWER_BASE)

    # sanitize overflow into [0, 1] (guard NaN/inf)
    of = overflow
    if not (of == of) or of in (float("inf"), float("-inf")):
        of = 1.0
    of = max(0.0, min(1.0, of))

    # Growth is monotone (>=1.0) but stronger while overlap remains high,
    # gentler as the placement legalizes -> lets wirelength dominate late.
    coef = 1.0 + (UPPER_PCOF - 1.0) * of
    mu = coef * base

    # Stagnation detection: if overflow has plateaued, push harder to escape.
    if len(overflow_history) >= 6:
        recent = overflow_history[-6:]
        progress = recent[0] - recent[-1]
        if progress < 0.002:
            mu *= 1.03
        elif progress < 0.0:  # overflow rising -> over-damped, accelerate
            mu *= 1.05

    # Gradient-aware brake: huge density gradients mean we are pushing too hard.
    if gradient_norm is not None and gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.02)

    # Near convergence, damp growth so fine wirelength tuning can proceed.
    if of < 0.10:
        mu = min(mu, 1.01)
    if of < 0.05:
        mu = min(mu, 1.005)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))