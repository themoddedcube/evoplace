import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    if ov != ov:  # NaN guard
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing: geometric decay blended with cosine ---
    # geometric gives smooth log-linear sweep from high to low gamma
    geo = gamma_high * (gamma_low / gamma_high) ** progress
    # cosine annealing damps the decay early, sharpens it late
    cos = gamma_low + 0.5 * (gamma_high - gamma_low) * (1.0 + math.cos(math.pi * progress))
    base = 0.5 * geo + 0.5 * cos

    # --- overflow coupling ---
    # When cells are still spread out (high overflow) keep gamma high for smooth
    # gradients; as the layout settles (low overflow) let gamma fall to refine HPWL.
    # Bounded multiplier in [0.7, 1.8] to avoid runaway gamma that diverges.
    overflow_factor = 0.7 + 1.1 * (ov ** 1.3)
    gamma = base * overflow_factor

    # --- HPWL feedback: react to plateau / divergence ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        finite = [h for h in recent if h == h and h not in (float("inf"), float("-inf"))]
        if len(finite) >= 2:
            prev = hpwl_history[-6] if len(hpwl_history) >= 6 else finite[0]
            best_recent = min(finite)
            # plateau: relative improvement stalled -> sharpen (lower gamma)
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85
            # divergence: HPWL climbing -> smooth gradients (raise gamma), bounded
            if finite[-1] > finite[0] * 1.02:
                gamma *= 1.25

    # --- end-game: force accurate, low-gamma regime for final refinement ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    if gamma != gamma:  # final NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))