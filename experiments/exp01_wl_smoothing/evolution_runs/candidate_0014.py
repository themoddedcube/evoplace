import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # Normalized training progress in [0, 1]
    t = iteration / max(1, total_iterations)

    # Clamp overflow to a sane range
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(1.0, max(0.0, of))

    # 1) Iteration-driven backbone: cosine anneal from high -> low gamma.
    #    High gamma early (smooth gradients, lets cells cluster/spread),
    #    low gamma late (accurate HPWL, fine-tuning).
    g_hi, g_lo = 8.0, 0.5
    backbone = g_lo + 0.5 * (g_hi - g_lo) * (1.0 + math.cos(math.pi * t))

    # 2) Overflow-adaptive multiplier (DREAMPlace-style log scaling).
    #    Keep gamma high while bins are still congested; let it fall as
    #    the layout legalizes. Centered so overflow~0.1 is neutral.
    adapt = 10.0 ** ((of - 0.1) * 1.2)

    gamma = backbone * adapt

    # 3) Plateau detection: if HPWL has stalled, sharpen the approximation
    #    (lower gamma) to chase finer wirelength gains.
    if hpwl_history is not None and len(hpwl_history) >= 6:
        recent = hpwl_history[-3:]
        prev = hpwl_history[-6:-3]
        avg_recent = sum(recent) / 3.0
        avg_prev = sum(prev) / 3.0
        if avg_prev > 0.0:
            rel_improve = (avg_prev - avg_recent) / avg_prev
            # Less than 0.1% improvement over the window => plateau
            if rel_improve < 1e-3:
                gamma *= 0.6

    # 4) Hard floor late in the schedule to guarantee an accurate final HPWL.
    if t > 0.9:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))