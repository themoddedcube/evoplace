import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # Normalized training progress in [0, 1]
    T = total_iterations if total_iterations and total_iterations > 0 else 1
    p = iteration / T
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0

    # Sanitize overflow (DREAMPlace overflow ~1.0 early, ->0 late)
    of = overflow
    if of != of:            # NaN guard
        of = 1.0
    if of < 0.0:
        of = 0.0
    elif of > 1.0:
        of = 1.0

    # ---- Base schedule: smooth high-gamma early, sharp low-gamma late ----
    # Exponential decay from 8.0 -> 0.5 in log-space (geometric interpolation).
    hi, lo = 8.0, 0.5
    base = hi * (lo / hi) ** p

    # ---- Overflow-adaptive coupling ----
    # While cells still overlap (high overflow) keep gamma smooth so gradients
    # stay stable; as overflow clears, let gamma fall for accurate HPWL.
    # Centered so of~0.1 (legal-ish) is neutral.
    of_factor = 10.0 ** (1.3 * (of - 0.1))

    gamma = base * of_factor

    # ---- HPWL-stagnation refinement ----
    # If wirelength has plateaued, nudge gamma lower to sharpen the WA-WL
    # approximation and escape the smoothed optimum.
    if hpwl_history and len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        prev = recent[0]
        ok = prev == prev and prev > 0.0
        if ok:
            rel = (prev - recent[-1]) / prev
            if rel == rel and rel < 1e-3:   # < 0.1% improvement over window
                gamma *= 0.85

    # Late-phase floor easing: ensure final iterations reach accurate regime.
    if p > 0.9:
        gamma = min(gamma, 1.0)

    if gamma != gamma:      # final NaN guard
        gamma = 1.0
    return min(50.0, max(0.01, gamma))