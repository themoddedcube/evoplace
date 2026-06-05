import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # Progress in [0, 1]
    T = max(1, total_iterations)
    t = min(max(iteration / T, 0.0), 1.0)

    # --- Base annealing in log-space: high gamma early -> low gamma late ---
    # Cosine schedule between log10(g_hi) and log10(g_lo) for smooth decay.
    g_hi, g_lo = 8.0, 0.5
    log_hi, log_lo = math.log10(g_hi), math.log10(g_lo)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))   # 1 at start -> 0 at end
    log_base = log_lo + (log_hi - log_lo) * cos_factor
    gamma = 10.0 ** log_base

    # --- Overflow-adaptive coupling ---
    # Placement quality is governed by density legalization (overflow), not just
    # iteration count. Keep gamma smooth while cells are still spread out
    # (high overflow), and only let it drop once density is resolving.
    ov = min(max(overflow, 0.0), 1.0)
    # Multiplicative boost that fades as overflow approaches the target band.
    ov_target = 0.10
    if ov > ov_target:
        # scale grows with excess overflow; capped to avoid runaway smoothing
        excess = (ov - ov_target) / (1.0 - ov_target)
        ov_boost = 1.0 + 4.0 * (excess ** 0.7)
    else:
        # below target: encourage sharper (lower) gamma for accurate HPWL
        ov_boost = max(0.5, ov / ov_target)
    gamma *= ov_boost

    # --- Stagnation detection: sharpen if HPWL has plateaued ---
    if hpwl_history and len(hpwl_history) >= 6:
        recent = hpwl_history[-3:]
        prev = hpwl_history[-6:-3]
        r = sum(recent) / 3.0
        p = sum(prev) / 3.0
        if p > 0.0:
            rel_improve = (p - r) / p
            # Near-zero or negative improvement -> reduce gamma to refine HPWL.
            if rel_improve < 1e-4:
                gamma *= 0.85

    # --- Late-phase guarantee: force accurate regime for final fine-tuning ---
    if t > 0.85:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))