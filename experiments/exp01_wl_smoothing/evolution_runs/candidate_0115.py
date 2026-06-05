import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule for differentiable global placement.

    Strategy: gamma is primarily a smooth (log-linear) function of overflow,
    which is the most reliable physical proxy for placement maturity. A mild
    progress term and plateau/divergence feedback refine it. High gamma while
    cells are spread (overflow high), low gamma for accurate HPWL once settled.
    """

    
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    
    
    
    
    ov_factor = ov ** 0.85
    gamma_ov = math.exp(math.log(gamma_low) + (math.log(gamma_high) - math.log(gamma_low)) * ov_factor)

    
    
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    
    
    w_ov = 0.70
    gamma = w_ov * gamma_ov + (1.0 - w_ov) * gamma_prog

    
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            
            
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            
            
            if window[-1] > window[0] * 1.015:
                gamma *= 1.30
            
            elif window[-1] < window[0] * 0.985:
                gamma *= 0.92

    
    
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.2)

    
    
    if ov < 0.08:
        gamma = min(gamma, 1.0)

    
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))