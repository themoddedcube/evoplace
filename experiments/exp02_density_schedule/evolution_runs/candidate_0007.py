def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ Overflow-adaptive density-weight schedule with hard bounds. """
    LOWER_PCOF = 0.95
    UPPER_PCOF = 1.05

    # --- baseline multiplicative growth, slowly relaxing over iterations ---
    decay = max(0.9999 ** float(iteration), 0.5)
    mu = 1.0 + (UPPER_PCOF - 1.0) * decay

    # --- overflow-adaptive correction (the main driver) ---
    if len(overflow_history) >= 2:
        prev = float(overflow_history[-2])
        delta = float(overflow) - prev
        ratio = delta / prev if prev > 1e-12 else 0.0

        if overflow < 0.08:
            # essentially legal: stop pushing density, let HPWL refine
            mu = LOWER_PCOF
        elif delta > 0.0:
            # overflow rising: spreading is losing ground -> push harder
            mu = UPPER_PCOF
        else:
            # overflow falling: scale growth by how fast it is dropping
            # fast drop  -> ease off (closer to LOWER_PCOF)
            # slow drop  -> keep pushing (closer to UPPER_PCOF)
            speed = min(1.0, -ratio)  # in [0,1]
            mu = UPPER_PCOF - (UPPER_PCOF - LOWER_PCOF) * speed
        mu = min(UPPER_PCOF, max(LOWER_PCOF, mu))

    # --- stability guard: damp growth if gradients are exploding ---
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4 and mu > 1.0:
            mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = current_lambda * mu

    # --- NaN/inf protection + required hard clamp to [0.01, 50.0] ---
    if not (new_lambda == new_lambda) or abs(new_lambda) == float("inf"):
        new_lambda = current_lambda
    return float(min(50.0, max(0.01, new_lambda)))