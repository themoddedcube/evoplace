def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplicative growth, annealed so early iters push density hard
    # and late iters settle (mirrors high-gamma-early -> low-gamma-late intuition).
    decay = max(0.9999 ** float(iteration), 0.98)
    mu = UPPER_PCOF * decay

    # Overflow-adaptive correction: speed up when spread is still poor,
    # slow down (or back off) once bins are nearly legal.
    if overflow > 0.10:
        mu *= 1.0 + min(0.5, 2.0 * (overflow - 0.10))   # accelerate density force
    elif overflow < 0.05:
        mu = LOWER_PCOF + (mu - LOWER_PCOF) * (overflow / 0.05)  # ease toward LOWER_PCOF

    # Stagnation / oscillation guard from history: if overflow stopped improving,
    # nudge lambda up to escape; if it is collapsing fast, damp to avoid overshoot.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[0] - recent[-1]          # positive => improving
        if delta < 1e-4 and overflow > 0.05:
            mu *= 1.10                           # break stagnation
        elif delta > 0.05:
            mu = min(mu, 1.02)                   # damp rapid collapse

    # Gradient-norm safety: huge gradients mean we are over-forcing; rein in.
    if gradient_norm > 0.0 and gradient_norm > 5.0:
        mu *= 0.97

    mu = min(max(mu, 0.90), 1.20)
    new_lambda = current_lambda * mu

    return min(max(new_lambda, 0.01), 50.0)