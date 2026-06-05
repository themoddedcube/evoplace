import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    t = iteration / max(1, total_iterations)
    of = min(1.0, max(0.0, overflow))

    # Base annealing: high gamma early (smooth, cells cluster) -> low gamma late (accurate HPWL).
    # Cosine schedule from g_hi to g_lo gives a gentle early hold and a soft late landing.
    g_hi, g_lo = 8.0, 0.5
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    base = g_lo + (g_hi - g_lo) * cos_factor

    # Overflow-adaptive multiplier (DREAMPlace-style): keep gamma high while bins are
    # congested, let it fall as the layout spreads out. Centered near overflow ~ 0.1.
    adapt = 10.0 ** (1.2 * (of - 0.1))
    gamma = base * adapt

    # Plateau-aware fine-tuning: once HPWL stops improving and density is acceptable,
    # drop gamma further so the WA-WL approximation tightens onto true HPWL.
    if len(hpwl_history) >= 5 and of < 0.15:
        recent = hpwl_history[-5:]
        ref = recent[0]
        if ref > 0.0 and abs(recent[-1] - ref) / ref < 1e-3:
            gamma *= 0.6

    # Hard floor late in placement to avoid over-smoothing during final convergence.
    if t > 0.9:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))