def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """Overflow-adaptive density-penalty schedule."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.003

    # Base geometric growth (DREAMPlace-style), decaying with iteration so the
    # penalty ramps hard early (cluster cells) and gently late (fine-tune HPWL).
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: when overflow stalls, push lambda harder; when it
    # is dropping fast, ease off so HPWL is not over-penalized.
    mu = base
    if overflow_history and len(overflow_history) >= 2:
        prev = float(overflow_history[-1])
        prev2 = float(overflow_history[-2])
        delta = prev2 - prev  # positive => overflow improving
        rel = delta / (prev2 + 1e-8)

        if rel < 0.005:
            # Stalled or worsening: accelerate the penalty.
            mu = base * (1.0 + min(0.10, (0.005 - rel) * 8.0))
        elif rel > 0.05:
            # Improving quickly: relax toward gentle growth.
            mu = max(LOWER_PCOF, base * (1.0 - min(0.04, (rel - 0.05) * 0.5)))

    # Late-stage fine-tuning: once nearly placed, hold lambda steady so the
    # solver can minimize HPWL without further density inflation.
    if overflow < 0.10:
        mu = min(mu, 1.0 + 0.5 * (overflow / 0.10))

    # Gradient-norm guard: if gradients explode, damp the growth for stability.
    if gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, LOWER_PCOF)

    new_lambda = current_lambda * mu
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)