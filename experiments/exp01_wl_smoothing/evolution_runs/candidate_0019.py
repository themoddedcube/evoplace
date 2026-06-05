import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """ ... """
    # Normalized progress in [0, 1]
    t = iteration / max(1, total_iterations)
    t = min(1.0, max(0.0, t))

    # Clamp overflow to a sane range (it can spike early)
    ov = min(1.0, max(0.0, overflow))

    # --- 1. Iteration-driven backbone: high gamma early, low gamma late ---
    # Cosine annealing between a high and low gamma gives a smooth, well-behaved
    # decay that keeps gradients smooth while cells are still clustering and
    # sharpens the HPWL approximation as the layout settles.
    g_hi, g_lo = 8.0, 0.5
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))      # 1 -> 0
    base = g_lo + (g_hi - g_lo) * cosine

    # --- 2. Overflow-adaptive modulation ---
    # When the placement is still spread out (high overflow) we want smoother
    # gradients (raise gamma); once density is resolved (low overflow) we trust
    # the iteration schedule and let gamma fall for accurate HPWL.
    # Multiplicative factor in roughly [0.6, 2.0].
    ov_factor = 0.6 + 1.4 * ov

    gamma = base * ov_factor

    # --- 3. Safety: react to divergence in the HPWL history ---
    # If recent HPWL is increasing (unstable / noisy gradients), bump gamma up
    # to re-smooth and stabilize before continuing to anneal.
    if hpwl_history is not None and len(hpwl_history) >= 3:
        recent = hpwl_history[-3:]
        if all(math.isfinite(x) for x in recent) and recent[-1] > recent[0] > 0:
            growth = recent[-1] / recent[0]
            if growth > 1.02:
                gamma *= min(2.0, growth)

    # --- 4. Floor near the end so gradients never vanish entirely ---
    if t > 0.9:
        gamma = max(gamma, 0.3)

    return min(50.0, max(0.01, gamma))