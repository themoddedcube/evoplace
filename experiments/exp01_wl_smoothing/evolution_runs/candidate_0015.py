import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # Progress in [0, 1]
    t = 0.0 if total_iterations <= 1 else iteration / float(total_iterations - 1)
    t = min(1.0, max(0.0, t))

    # Base schedule: high gamma early (smooth, cluster cells) -> low gamma late
    # (accurate HPWL, fine-tuning). Cosine annealing on a log scale gives a
    # smooth high->low transition that lingers high early and decays gently.
    gamma_hi = 8.0
    gamma_lo = 0.5
    log_hi = math.log(gamma_hi)
    log_lo = math.log(gamma_lo)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 at start -> 0 at end
    log_gamma = log_lo + (log_hi - log_lo) * cos_factor
    gamma = math.exp(log_gamma)

    # Overflow-adaptive coupling: gamma should track how clustered/spread the
    # layout actually is, not just the iteration counter. High overflow means
    # cells still overlap heavily -> keep gradients smooth (raise gamma). Low
    # overflow means the layout is legal-ish -> sharpen toward true HPWL.
    of = min(1.0, max(0.0, overflow))
    # Map overflow into a multiplicative factor centered at 1.0.
    # of ~ 0.9 (very congested) -> boost; of ~ 0.1 (nearly legal) -> shrink.
    overflow_factor = math.exp(1.4 * (of - 0.5))
    gamma *= overflow_factor

    # Plateau detection: if HPWL has stopped improving, sharpen gamma to escape
    # the over-smoothed approximation and pursue accurate wirelength.
    if hpwl_history is not None and len(hpwl_history) >= 6:
        recent = hpwl_history[-3:]
        prev = hpwl_history[-6:-3]
        recent_avg = sum(recent) / len(recent)
        prev_avg = sum(prev) / len(prev)
        if prev_avg > 0.0:
            rel_improve = (prev_avg - recent_avg) / abs(prev_avg)
            if rel_improve < 1e-4:  # stalled or worsening
                gamma *= 0.7

    # Late-stage floor relaxation: force accuracy in the final phase.
    if t > 0.85:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))