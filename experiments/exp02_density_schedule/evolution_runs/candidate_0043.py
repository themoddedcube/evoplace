def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ ... """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # DREAMPlace-style decaying base multiplier: aggressive early, gentle late
    base = max(0.9999 ** float(iteration), 0.98)

    of = overflow if overflow == overflow else 1.0   # NaN guard
    of = min(max(of, 0.0), 1.0)

    # Progress signal: how fast is overflow clearing?
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        delta = prev - overflow_history[-1]          # >0 means improving
    else:
        delta = 0.0

    # Stall detector: when overflow plateaus, push density weight harder so
    # cells finish spreading; when it is dropping fast, ease off and let the
    # wirelength term refine the placement.
    if delta <= 0.0:
        stall = 1.0
    else:
        stall = max(0.0, 1.0 - delta * 20.0)

    # Overflow-adaptive blend: high overflow (clustered) -> grow faster,
    # low overflow (spread, near-converged) -> back toward neutral for HPWL.
    blend = 0.5 * of + 0.5 * stall
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * blend

    mu = pcof * base

    # Late-stage fine-tuning: once mostly spread, relax weight growth so the
    # optimizer can trade density slack for shorter wirelength.
    if of < 0.10:
        mu = min(mu, 1.0 + (mu - 1.0) * (of / 0.10))

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))