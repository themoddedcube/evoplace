def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Annealed base multiplier: aggressive growth early, gentler late.
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow trend from history (smoothed over last few steps if available).
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = (recent[-1] - recent[0]) / 2.0
    elif len(overflow_history) >= 2:
        delta = overflow_history[-1] - overflow_history[-2]
    else:
        delta = 0.0

    # Adaptive coefficient: ramp lambda harder when overflow stalls or rises,
    # ease off when density is already improving quickly.
    if delta >= 0.0:
        coef = UPPER_PCOF
    else:
        rate = min(1.0, (-delta) / 0.02)
        coef = UPPER_PCOF + (LOWER_PCOF - UPPER_PCOF) * rate

    mu = coef * base

    # Late-stage fine-tuning: once cells are nearly legal, stop forcing the
    # penalty up so the placer can refine HPWL with accurate gradients.
    if overflow < 0.10:
        mu = min(mu, 1.0 + 5.0 * overflow)

    # Damp updates when gradients are exploding to avoid overshoot.
    if gradient_norm > 0.0 and gradient_norm > 5.0:
        mu = 1.0 + (mu - 1.0) * (5.0 / gradient_norm)

    new_lambda = current_lambda * mu
    return float(min(50.0, max(0.01, new_lambda)))