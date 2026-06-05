import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # Normalized progress in [0, 1]
    T = max(1, total_iterations)
    t = min(max(iteration, 0), T) / T

    # Sanitize overflow
    ovfl = overflow if (overflow is not None and overflow == overflow) else 1.0
    ovfl = min(1.0, max(0.0, ovfl))

    # 1) Base schedule: cosine-annealed exponential decay from high -> low gamma.
    #    High gamma early (smooth gradients, cells cluster), low gamma late
    #    (accurate HPWL, fine placement). Decay in log-space for smoothness.
    g_hi, g_lo = 8.0, 0.5
    cos = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
    log_gamma = math.log(g_lo) + (math.log(g_hi) - math.log(g_lo)) * cos
    gamma = math.exp(log_gamma)

    # 2) Overflow-adaptive coupling: placement quality is governed by overflow,
    #    not just iteration count. While bins are congested keep gamma elevated
    #    so gradients stay smooth; relax as overflow drops toward target (~0.1).
    target_ovfl = 0.10
    if ovfl > target_ovfl:
        # excess overflow in [0, ~0.9] -> multiplicative boost up to ~2x
        excess = (ovfl - target_ovfl) / (1.0 - target_ovfl)
        gamma *= 1.0 + 1.0 * max(0.0, min(1.0, excess))
    else:
        # below target: push toward accurate (low) gamma for fine-tuning
        gamma *= 0.6 + 0.4 * (ovfl / target_ovfl)

    # 3) Plateau detection: if HPWL has stopped improving, lower gamma a touch
    #    to sharpen the wirelength approximation and escape the plateau.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        try:
            best = min(recent)
            last = recent[-1]
            if best > 0 and (last - best) / best < 1e-4:
                gamma *= 0.85
        except (TypeError, ValueError):
            pass

    # Guarantee monotone-friendly late stage: hard cap that decays with progress
    late_cap = g_hi * (1.0 - 0.5 * t) + 0.5
    gamma = min(gamma, late_cap)

    return min(50.0, max(0.01, gamma))