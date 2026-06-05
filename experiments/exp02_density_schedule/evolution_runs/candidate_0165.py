def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """Overflow-adaptive density-weight schedule with divergence guards.

    Grows the density penalty multiplicatively (DREAMPlace style) but
    modulates the step by current overflow and its trend, and hard-clamps
    the result so a runaway weight can never blow HPWL up to inf.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # --- sanitize inputs (NaN/inf -> safe defaults) ---
    of = overflow
    if not (of == of) or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    gn = gradient_norm
    if not (gn == gn):
        gn = 0.0

    cur = current_lambda
    if not (cur == cur) or cur <= 0.0:
        cur = 0.01

    # --- base multiplicative growth, annealing with iteration ---
    base_mu = max(0.9999 ** float(iteration), 0.98)

    # --- overflow-adaptive coefficient: push hard while bins are full,
    #     ease toward 1.0 (and below) as the layout legalizes ---
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # --- trend: if overflow has stalled while still high, nudge weight up;
    #     if overflow is already collapsing, relax to let HPWL fine-tune ---
    if overflow_history:
        recent = [h for h in overflow_history[-3:] if h == h]
        if len(recent) >= 2:
            improvement = recent[0] - recent[-1]   # positive => improving
            if improvement <= 1e-4 and of > 0.10:
                coef = min(coef * 1.03, UPPER_PCOF)
            elif improvement > 1e-2 and of < 0.10:
                coef = max(coef * 0.98, LOWER_PCOF)

    mu = base_mu * coef

    # --- gradient safety: damp increases when gradients are exploding ---
    if gn > 1e3:
        mu = min(mu, 1.0)

    new_lambda = cur * mu

    # --- final divergence guard + valid range clamp ---
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cur
    return float(min(max(new_lambda, 0.01), 50.0))