import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma with iteration annealing and stagnation kick."""

    # Guard against degenerate inputs.
    total = max(1, total_iterations)
    t = min(1.0, max(0.0, iteration / total))
    ov = overflow if (overflow == overflow) else 1.0  # NaN -> treat as full
    ov = min(1.0, max(0.0, ov))

    # --- Primary driver: overflow-adaptive exponential (DREAMPlace-style) ---
    # High overflow (cells overlapping) -> large gamma for smooth, global gradients.
    # Low overflow (placement legal) -> small gamma for accurate HPWL.
    # Maps ov in [0,1] to gamma in ~[0.5, 20] on a log scale.
    gamma_ov = 10.0 ** (1.301 * ov - 0.301)  # ov=1 -> ~10, ov=0 -> ~0.5

    # --- Secondary driver: cosine annealing over iterations ---
    # Provides a smooth high->low decay even if overflow plateaus, so the tail
    # always sharpens HPWL accuracy near the end of the run.
    gamma_hi, gamma_lo = 8.0, 0.5
    cos = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
    gamma_anneal = gamma_lo + (gamma_hi - gamma_lo) * cos

    # Blend: early on trust overflow (global structure), late trust the anneal.
    w = t  # 0 early -> 1 late
    gamma = (1.0 - w) * gamma_ov + w * min(gamma_ov, gamma_anneal)

    # --- Stagnation kick: if HPWL stops improving, briefly raise gamma to
    # smooth the landscape and escape a poor local configuration. ---
    if len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        finite = [h for h in recent if h == h and h not in (float("inf"), float("-inf"))]
        if len(finite) >= 2:
            improvement = (finite[0] - finite[-1]) / (abs(finite[0]) + 1e-12)
            if improvement < 1e-4 and ov > 0.1:
                gamma *= 1.5

    # Never collapse gamma while overflow is still high (avoids noisy gradients
    # tearing apart an un-legalized layout, which can diverge HPWL to inf).
    floor = 0.5 + 3.5 * ov
    gamma = max(gamma, floor)

    return min(50.0, max(0.01, gamma))