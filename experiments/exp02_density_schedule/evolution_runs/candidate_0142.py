def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """Overflow-adaptive density-weight (lambda) multiplier."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # DREAMPlace-style baseline: strong early growth that gently decays.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Measure recent overflow progress (smoothed over last few steps).
    delta = 0.0
    if overflow_history and len(overflow_history) >= 2:
        window = overflow_history[-4:]
        delta = (window[-1] - window[0]) / float(len(window) - 1)

    # Adapt the multiplier to how the placement is actually converging.
    if delta > 0.0:
        # Overflow rising/stagnant: push the density penalty harder.
        accel = 1.0 + min(0.04, 12.0 * delta)
    elif delta < 0.0:
        # Overflow falling well: relax growth so HPWL can settle.
        accel = max(0.80, 1.0 + 6.0 * delta)
    else:
        accel = 1.0

    # Fine-tune phase: once nearly legal, taper toward neutral growth
    # so wirelength is optimized instead of over-spreading.
    if overflow < 0.10:
        taper = 0.5 + 5.0 * max(overflow, 0.0)   # in [0.5, 1.0]
        taper = min(1.0, taper)
    else:
        taper = 1.0

    # Mild damping when gradients explode (avoids destructive steps).
    if gradient_norm > 0.0:
        grad_damp = 1.0 if gradient_norm < 1.0 else max(0.90, 1.0 / (gradient_norm ** 0.05))
    else:
        grad_damp = 1.0

    mu = base * accel * taper * grad_damp
    mu = max(LOWER_PCOF * 0.95, mu)

    new_lambda = current_lambda * mu
    return float(min(50.0, max(0.01, new_lambda)))