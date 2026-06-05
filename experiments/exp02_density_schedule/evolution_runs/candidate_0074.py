def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Sanitize inputs
    of = overflow if overflow is not None else 1.0
    if of < 0.0:
        of = 0.0
    elif of > 1.0:
        of = 1.0

    # Annealed growth envelope (matches the original's decaying step)
    base = 0.9999 ** float(iteration)
    if base < 0.98:
        base = 0.98

    # Overflow-adaptive multiplier: grow lambda hard while many bins are
    # overfull (cells still need to spread), ease toward 1 as it legalizes.
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of
    mu *= base

    # Trend from history: positive = overflow still dropping (improving).
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        try:
            trend = float(overflow_history[-2]) - float(overflow_history[-1])
        except (TypeError, ValueError):
            trend = 0.0

    # Stalled but still congested -> push harder to break the plateau.
    if trend <= 1e-4 and of > 0.10:
        mu *= 1.02
    # Legalizing quickly while overflow is already low -> back off so the
    # density penalty doesn't overwhelm wirelength in the fine-tuning phase.
    elif of < 0.10 and trend > 1e-3:
        mu *= 0.99

    # Gradient safeguard: temper the step if gradients blow up.
    if gradient_norm is not None and gradient_norm > 1e3:
        mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = current_lambda * mu

    # Hard clamp to the allowed range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)