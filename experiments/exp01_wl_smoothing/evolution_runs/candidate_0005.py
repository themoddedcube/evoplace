import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    gamma_max = 8.0
    gamma_min = 0.5
    t = iteration / max(total_iterations - 1, 1)

    ov = max(0.0, min(1.0, overflow))

    # Overflow-adaptive: γ tracks placement legality state directly.
    # High overflow (cells clustered) → high γ (smooth WA-WL, global flow).
    # Low overflow (near-legal) → low γ (accurate HPWL, fine-tuning).
    ov_gamma = gamma_min + (gamma_max - gamma_min) * math.exp(-2.5 * (1.0 - ov))

    # Cosine annealing: smooth time-based monotone reference.
    cos_gamma = gamma_min + (gamma_max - gamma_min) * 0.5 * (1.0 + math.cos(math.pi * t))

    # When overflow is high → trust overflow signal.
    # When overflow is low → take conservative minimum to sharpen HPWL accuracy.
    alpha = min(1.0, ov * 1.5)
    gamma = alpha * ov_gamma + (1.0 - alpha) * min(ov_gamma, cos_gamma)

    # Late-stage stagnation: if HPWL not improving with low overflow,
    # push gamma down to sharpen gradients and break the plateau.
    if len(hpwl_history) >= 4 and t > 0.6 and ov < 0.3:
        h = hpwl_history
        if h[-4] > 0 and (h[-4] - h[-1]) / h[-4] < 0.002:
            gamma = gamma_min + (gamma - gamma_min) * 0.75

    return max(gamma_min, min(20.0, gamma))