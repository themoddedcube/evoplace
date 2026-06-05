def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    LOWER_PCOF = 0.95
    UPPER_PCOF = 1.05

    # Base multiplicative growth (DREAMPlace subgradient style):
    # density weight ramps up smoothly, decaying the step as we anneal.
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive speed: when cells are still spreading well
    # (overflow falling) push the density weight up faster; when the
    # spread stalls or worsens, hold back so gradients stay stable.
    if len(overflow_history) >= 2:
        delta = overflow - overflow_history[-2]
        if delta < 0.0:
            speed = UPPER_PCOF
        else:
            # stalled/worsening -> temper growth, scaled by remaining congestion
            speed = LOWER_PCOF + 0.10 * max(min(overflow, 1.0), 0.0)
    else:
        speed = UPPER_PCOF

    mu = speed * base

    # NaN / non-positive guard on the multiplier
    if mu != mu or mu <= 0.0:
        mu = 1.0
    # Keep per-step change bounded to avoid runaway / divergence (inf HPWL)
    mu = min(max(mu, 0.90), 1.10)

    new_lambda = current_lambda * mu

    # Final NaN / inf guard and hard clamp to the valid range.
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))