import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware log-cosine gamma annealing for DREAMPlace WA-WL.

    High gamma early (smooth gradients, cells cluster) decaying to low gamma
    late (accurate HPWL). Decay is primarily driven by physical convergence
    (overflow) and modulated by iteration progress, with gentle, bounded
    plateau/divergence responses to avoid the oscillation that destabilizes
    the optimizer.
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

    
    
    
    
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    iter_base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    
    
    
    ov_base = gamma_low * (gamma_high / gamma_low) ** (ov ** 0.85)

    
    
    
    w_ov = 0.65
    gamma = (iter_base ** (1.0 - w_ov)) * (ov_base ** w_ov)

    
    
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev_best = min(recent[:-1]) if len(recent) >= 6 else window[0]

            
            
            if prev_best > 0 and (prev_best - best_recent) / prev_best < 1e-3:
                gamma *= 0.90

            
            
            if window[-1] > window[0] * 1.01:
                gamma *= 1.15
            
            elif window[-1] < window[0] * 0.99:
                gamma *= 0.97

    
    
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.8
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))