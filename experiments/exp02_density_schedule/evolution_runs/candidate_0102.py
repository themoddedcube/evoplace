def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """Overflow-adaptive density-penalty multiplier, clamped to the legal range."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    it = float(iteration)
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = 0.0 if of < 0.0 else (1.0 if of > 1.0 else of)

    # Geometric growth that anneals with iterations so lambda does not run away.
    anneal = max(0.9995 ** it, 0.985)

    # Grow the penalty faster while many bins are over-dense, ease off as the
    # layout spreads (low overflow -> mu near LOWER_PCOF -> gentle fine-tuning).
    mu = (LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of) * anneal

    # Use overflow trend: if spreading has stalled, push harder; if it is
    # improving steadily, relax to let HPWL settle.
    if overflow_history and len(overflow_history) >= 2:
        delta = float(overflow_history[-1]) - float(overflow_history[-2])
        if delta > 1e-4:          # overflow rising -> need more spreading force
            mu *= 1.02
        elif delta < -1e-3:       # spreading well -> back off
            mu *= 0.99

    # Damp the update if gradients are exploding to keep optimization stable.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Guard against NaN/inf and enforce the required output range.
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = current_lambda
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)