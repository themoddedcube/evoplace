def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    TARGET_OVERFLOW = 0.10

    # Sanitize inputs (guard against NaN/inf that cause divergence)
    of = overflow
    if of != of or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    # Base multiplicative growth, gently annealed with iteration so the
    # density weight ramps hard early and stabilizes late for fine HPWL.
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: push lambda harder while many bins are over-dense,
    # ease toward neutral as overflow approaches target so cells settle.
    adapt = 1.0 + (of - TARGET_OVERFLOW)

    # Stall detection: if overflow has stopped improving, nudge growth up.
    if isinstance(overflow_history, list) and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] >= recent[0] - 1e-4:
            adapt *= 1.03

    mu = UPPER_PCOF * base * adapt

    # Clamp the per-step multiplier — the key fix against runaway divergence.
    mu = min(max(mu, LOWER_PCOF), 1.10)

    # Gradient safeguard: if gradients explode, do not amplify lambda.
    gn = gradient_norm
    if gn != gn or gn in (float("inf"), float("-inf")) or gn > 1e6:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = current_lambda

    return float(min(max(new_lambda, 0.01), 50.0))