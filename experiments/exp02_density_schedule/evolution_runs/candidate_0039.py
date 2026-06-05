def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplicative growth, decaying with iteration (DREAMPlace-style).
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    of = overflow if overflow == overflow else 1.0   # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Overflow-adaptive: push the density penalty hard while cells are
    # still spread out, and ease off as bins clear so HPWL can settle.
    mu = base * (0.5 + of)

    # Trend control from overflow history.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        cur = overflow_history[-1]
        if prev > 1e-12:
            ratio = cur / prev
            if ratio > 0.995:        # stagnating -> increase penalty
                mu *= 1.10
            elif ratio < 0.95:       # clearing fast -> protect wirelength
                mu *= LOWER_PCOF

    # Gradient safeguard: damp growth if gradients are exploding.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.0)
        elif gradient_norm > 1e3:
            mu = min(mu, 1.02)

    # Near convergence (overflow essentially cleared), stop inflating lambda.
    if of < 0.08:
        mu = min(mu, 1.005)

    new_lambda = current_lambda * mu

    if new_lambda != new_lambda:     # NaN fallback
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))