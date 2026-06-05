import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma schedule for differentiable global placement.

    Anneals gamma from high (smooth gradients, coarse clustering) to low
    (accurate HPWL, fine placement). The decay is primarily driven by
    physical overflow rather than raw iteration count, because overflow is
    a far more reliable indicator of how spread-out the cells still are.
    Iteration progress is used only as a gentle floor/ceiling guard so the
    schedule keeps moving even when overflow stalls.
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
    log_high = math.log(gamma_high)
    log_low = math.log(gamma_low)

    
    
    
    
    
    ov_drive = ov ** 0.65
    t_drive = 0.5 - 0.5 * math.cos(math.pi * progress)   

    
    
    decay = 0.70 * (1.0 - ov_drive) + 0.30 * t_drive
    decay = min(1.0, max(0.0, decay))

    
    log_gamma = log_high + (log_low - log_high) * decay
    gamma = math.exp(log_gamma)

    
    
    if hpwl_history:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0 and h < float("inf")]
        if len(recent) >= 5:
            window = recent[-5:]
            first = window[0]
            last = window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            
            if first > 0 and last > first * 1.02:
                gamma *= 1.30
            
            elif first > 0 and last < first * 0.97:
                gamma *= 0.92

    
    
    if progress > 0.90:
        ceil = 1.2 if ov > 0.08 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.75:
        gamma = min(gamma, 2.2 if ov > 0.08 else 1.2)
    elif progress > 0.55:
        gamma = min(gamma, 4.0)

    
    floor = gamma_low if ov < 0.05 else (gamma_low + 0.3 * (ov ** 1.5))
    gamma = max(gamma, floor)

    
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))