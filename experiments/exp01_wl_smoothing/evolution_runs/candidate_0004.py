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
    overflow_clamped = max(0.0, min(1.0, overflow))

    # Overflow-coupled decay: gamma tracks spreading progress directly
    # sqrt gives faster initial drop as cells begin to spread
    gamma_overflow = gamma_min + (gamma_max - gamma_min) * math.sqrt(overflow_clamped)

    # Exponential time decay as a ceiling — ensures progress even if overflow stalls
    gamma_time = gamma_min + (gamma_max - gamma_min) * math.exp(-3.5 * t)

    # Follow whichever signal decays faster
    gamma = min(gamma_overflow, gamma_time)

    # If HPWL is stagnating, tighten gamma to sharpen gradients
    if len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        if recent[-1] >= recent[0] * 0.999:
            gamma = max(gamma * 0.85, gamma_min)

    return max(gamma_min, min(20.0, gamma))