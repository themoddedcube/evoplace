def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """Overflow-adaptive gamma anneal: smooth early, accurate late."""
    GAMMA_MIN = 0.5
    GAMMA_MAX = 8.0

    # Sanitize overflow into [0, 1] (guard NaN/Inf).
    ov = overflow
    if not (ov == ov) or ov in (float("inf"), float("-inf")):
        ov = 1.0
    if ov < 0.0:
        ov = 0.0
    elif ov > 1.0:
        ov = 1.0

    # Overflow -> gamma: congested bins keep gamma high (smooth gradients,
    # cells cluster); as the layout spreads, gamma falls (sharp HPWL).
    gamma = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * (ov ** 0.5)

    # Iteration anneal floor so gamma keeps dropping even if overflow plateaus.
    decay = 0.999 ** float(iteration)
    gamma = GAMMA_MIN + (gamma - GAMMA_MIN) * decay

    # If overflow has stalled, sharpen the approximation further.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] >= recent[0] - 1e-4:
            gamma *= 0.9

    # Clamp to the allowed return range.
    if not (gamma == gamma):
        gamma = GAMMA_MIN
    if gamma < 0.01:
        gamma = 0.01
    elif gamma > 50.0:
        gamma = 50.0
    return gamma