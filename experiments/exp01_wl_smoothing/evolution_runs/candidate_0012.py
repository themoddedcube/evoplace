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

    # Clamp overflow to a sane range
    ov = overflow if overflow == overflow else 1.0  # guard against NaN
    ov = min(1.0, max(0.0, ov))

    # --- Base schedule: high gamma early -> low gamma late ---
    # Cosine annealing between gamma_hi and gamma_lo over the run.
    gamma_hi = 8.0
    gamma_lo = 0.5
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 at start -> 0 at end
    base = gamma_lo + (gamma_hi - gamma_lo) * cos_factor

    # --- Overflow-adaptive boost ---
    # When cells are still spread out (high overflow), keep gradients smooth
    # by raising gamma. As overflow collapses toward the target, let gamma fall
    # so HPWL is approximated accurately for fine-tuning.
    target_ov = 0.1
    if ov > target_ov:
        # Smoothly scale up by up to ~3x while overflow is high.
        excess = (ov - target_ov) / (1.0 - target_ov)  # in [0, 1]
        overflow_mult = 1.0 + 2.0 * excess
    else:
        # Below target: aggressively reduce gamma for accurate HPWL.
        overflow_mult = max(0.25, ov / target_ov)

    gamma = base * overflow_mult

    # --- HPWL-history stagnation relief ---
    # If recent HPWL has plateaued, nudge gamma down to sharpen the objective.
    if hpwl_history and len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        prev = recent[0]
        last = recent[-1]
        if prev > 0 and abs(last - prev) / prev < 1e-3:
            gamma *= 0.7

    return min(50.0, max(0.01, gamma))