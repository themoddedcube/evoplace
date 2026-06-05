def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ Overflow-adaptive multiplicative density-weight schedule with clamping. """
    # --- sanitize inputs ---
    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)
    cl = current_lambda if current_lambda == current_lambda else 1.0
    if cl <= 0.0:
        cl = 0.01

    # --- base annealing factor (DREAMPlace-style, decaying with iteration) ---
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    base = max(0.9999 ** float(iteration), 0.98)

    # --- overflow-adaptive ramp ---
    # High overflow (cells still spread/over-dense) -> push lambda up faster to
    # cluster/compact. Low overflow (legalizable) -> ease off so HPWL gradients
    # dominate and we fine-tune instead of over-penalizing.
    if of > 0.10:
        ramp = UPPER_PCOF * base
    else:
        # near-converged density: interpolate factor toward ~1.0
        t = of / 0.10                                       # in [0, 1]
        ramp = (LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * t) * base

    # --- stall detection: if overflow has plateaued, nudge harder ---
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        spread = max(recent) - min(recent)
        if spread < 1e-3 and of > 0.10:
            ramp *= 1.02

    # --- gradient guard: damp growth if gradients are exploding ---
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e6:
        ramp = min(ramp, 1.0)

    new_lambda = cl * ramp

    # --- hard clamp to required range ---
    if new_lambda != new_lambda:                            # NaN
        new_lambda = 1.0
    return float(min(max(new_lambda, 0.01), 50.0))