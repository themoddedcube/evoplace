def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    GAMMA_HIGH = 8.0
    GAMMA_LOW = 0.5

    # Smooth overflow with recent history to damp gradient noise.
    if overflow_history:
        recent = overflow_history[-3:]
        ov = 0.6 * (sum(recent) / len(recent)) + 0.4 * overflow
    else:
        ov = overflow
    ov = min(max(ov, 0.0), 1.0)

    # Overflow-adaptive: cells clustered (high overflow) -> smooth high gamma;
    # cells spread (low overflow) -> accurate low gamma. Log-uniform mapping.
    gamma = GAMMA_LOW * (GAMMA_HIGH / GAMMA_LOW) ** ov

    # Iteration floor: guarantee fine-tuning late even if overflow plateaus.
    anneal = max(0.99 ** float(iteration), 0.2)
    gamma = GAMMA_LOW + (gamma - GAMMA_LOW) * anneal

    # If overflow is essentially resolved, push toward the accurate regime.
    if ov < 0.08:
        gamma = min(gamma, GAMMA_LOW + 0.5 * ov / 0.08)

    # Stabilize against exploding gradients by softening gamma slightly.
    if gradient_norm > 0.0 and gradient_norm > 5.0:
        gamma = gamma * 1.1

    return float(min(max(gamma, 0.01), 50.0))