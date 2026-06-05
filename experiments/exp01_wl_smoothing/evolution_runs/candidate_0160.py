import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for differentiable global placement.

    Anneals gamma from high (smooth gradients, cells clustering) to low
    (accurate HPWL, fine placement) over the run, modulated by the live
    overflow signal and tempered by HPWL-history plateau/divergence feedback.
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
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    
    
    
    
    
    ov_eff = ov ** 1.3
    ov_mult = 0.50 + 1.65 * ov_eff
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.6)
    gamma = 0.55 * base * ov_mult + 0.45 * ov_add

    
    
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            
            
            if first > 0:
                rel_improve = (first - best_recent) / first
            else:
                rel_improve = 1.0

            
            
            
            if prev > 0 and (prev - best_recent) / prev < 1e-3 and ov < 0.30:
                gamma *= 0.80

            
            
            if last > first * 1.02:
                gamma *= 1.30
            
            elif last < first * 0.985 and rel_improve > 5e-3:
                gamma *= 0.92

    
    
    
    if progress > 0.85:
        ceil = 1.4 if ov > 0.10 else 0.65
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.4 if ov > 0.10 else 1.4)
    elif progress > 0.50:
        gamma = min(gamma, 4.0 if ov > 0.15 else 2.5)

    
    
    if progress < 0.10:
        gamma = max(gamma, 4.0)

    
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))