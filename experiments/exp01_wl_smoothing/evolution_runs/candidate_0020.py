import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # Progress in [0, 1]
    if total_iterations <= 1:
        progress = 1.0
    else:
        progress = iteration / float(total_iterations - 1)
    progress = min(1.0, max(0.0, progress))

    # Clamp overflow defensively
    of = min(1.0, max(0.0, overflow))

    # --- Base schedule: exponential decay high -> low over iterations ---
    # gamma goes from ~8.0 (smooth, cluster) down to ~0.5 (accurate HPWL)
    g_hi, g_lo = 8.0, 0.5
    base = g_hi * (g_lo / g_hi) ** progress

    # --- Overflow-adaptive term (DREAMPlace style) ---
    # While density is still high, keep gradients smooth by boosting gamma.
    # As overflow collapses toward 0, this term vanishes and HPWL accuracy wins.
    of_factor = 10.0 ** (2.0 * (of - 0.1))
    of_factor = min(8.0, max(0.25, of_factor))

    gamma = base * of_factor

    # --- Cosine micro-annealing for the final fine-tuning window ---
    # Smoothly push toward the accurate (low-gamma) regime near the end.
    if progress > 0.75:
        tail = (progress - 0.75) / 0.25
        anneal = 0.5 * (1.0 + math.cos(math.pi * tail))  # 1 -> 0
        gamma = g_lo + (gamma - g_lo) * anneal

    # --- Plateau detection: if HPWL stalls, sharpen (lower gamma) ---
    if len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        prev = recent[0]
        if prev > 0:
            rel_change = abs(recent[-1] - prev) / abs(prev)
            if rel_change < 1e-3:
                gamma *= 0.7

    return min(50.0, max(0.01, gamma))