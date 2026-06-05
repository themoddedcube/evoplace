def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """Overflow-adaptive density-weight escalation with hard bounds.

    Grows lambda geometrically (DREAMPlace-style) but modulates the step by
    the current overflow and its recent trend, then clamps the result so a
    runaway multiplier can never drive the optimization to inf.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric growth, decaying toward a floor so early iterations push
    # hardest (cells still need to spread) and later ones ease off.
    base = max(0.9999 ** float(iteration), 0.98)
    mu = UPPER_PCOF * base

    # Overflow gating: hammer the penalty while bins are very congested,
    # release it once the placement is nearly legal so HPWL can settle.
    if overflow > 0.5:
        mu = UPPER_PCOF
    elif overflow < 0.10:
        mu = min(mu, 1.0)
    elif overflow < 0.05:
        mu = min(mu, LOWER_PCOF)  # back off to let wirelength fine-tune

    # Trend correction: if overflow has stalled (not decreasing), nudge the
    # penalty up; if it is dropping fast, avoid over-penalizing.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[0] - recent[-1]
        if delta < 1e-3 and overflow > 0.10:
            mu *= 1.03                      # break stagnation
        elif delta > 0.05:
            mu = min(mu, 1.0)               # already converging, hold steady

    # Gradient safety: if gradients are exploding, do not amplify the penalty.
    if gradient_norm > 0.0 and not (gradient_norm < 1e6):
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Hard clamp to the legal range — guarantees a finite, in-bounds return.
    return float(min(max(new_lambda, 0.01), 50.0))