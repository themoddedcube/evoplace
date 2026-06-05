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

    # --- Base annealing: high gamma early -> low gamma late ---
    # Exponential decay from g_hi to g_lo over the run.
    g_hi, g_lo = 8.0, 0.5
    base = g_hi * (g_lo / g_hi) ** t  # geometric interpolation

    # --- Overflow-adaptive term ---
    # While cells are still poorly spread (high overflow) we want smoother
    # gradients (higher gamma); as overflow collapses we sharpen toward the
    # true HPWL. Clamp overflow to a sane range first.
    ov = overflow if overflow == overflow else 1.0  # guard NaN
    ov = min(1.5, max(0.0, ov))
    # Maps ov in [0.1, 1.0] -> multiplier in ~[0.7, 1.6]
    ov_mult = 0.7 + 0.9 * min(1.0, ov / 0.9)

    gamma = base * ov_mult

    # --- Cosine fine-tuning floor in the final phase ---
    # In the last 20% of iterations, ease gamma down smoothly toward g_lo
    # so the placement settles on an accurate wirelength estimate.
    if t > 0.8:
        tau = (t - 0.8) / 0.2  # 0 -> 1 over the tail
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * tau))  # 1 -> 0
        gamma = g_lo + (gamma - g_lo) * cos_factor

    # --- Stagnation detection: if HPWL has plateaued, sharpen sooner ---
    if hpwl_history and len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        lo, hi = min(recent), max(recent)
        denom = abs(hi) + 1e-12
        if (hi - lo) / denom < 1e-3:  # < 0.1% spread -> plateau
            gamma *= 0.8

    return min(50.0, max(0.01, gamma))