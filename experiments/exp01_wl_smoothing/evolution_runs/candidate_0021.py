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

    # Sanitize overflow into [0, 1]
    try:
        ov = float(overflow)
    except (TypeError, ValueError):
        ov = 0.5
    if not math.isfinite(ov):
        ov = 0.5
    ov = min(1.0, max(0.0, ov))

    # --- Base schedule: high gamma early -> low gamma late ---
    # Exponential decay in log-space between gamma_hi and gamma_lo.
    gamma_hi = 8.0
    gamma_lo = 0.5
    # Cosine annealing of the log-gamma gives a smooth, slow-then-fast taper
    # that keeps cells clustered early and sharpens HPWL accuracy late.
    cos_anneal = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
    log_base = math.log(gamma_lo) + (math.log(gamma_hi) - math.log(gamma_lo)) * cos_anneal
    gamma = math.exp(log_base)

    # --- Overflow-adaptive coupling ---
    # When overflow is still high, density hasn't resolved: keep gradients smooth
    # by biasing gamma upward. When overflow is low, trust the geometry and sharpen.
    # Multiplicative factor in roughly [0.6, 1.8].
    overflow_factor = 0.6 + 1.2 * ov
    gamma *= overflow_factor

    # --- Plateau detection on HPWL history ---
    # If HPWL has stagnated, nudge gamma down to escape the smooth-approx regime
    # and recover wirelength accuracy.
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-4:]
                  if isinstance(h, (int, float)) and math.isfinite(h)]
        if len(recent) >= 4 and recent[0] != 0.0:
            rel_change = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-12)
            if rel_change < 1e-3:
                gamma *= 0.85

    # Guard against non-finite results before clamping.
    if not math.isfinite(gamma):
        gamma = 1.0

    return min(50.0, max(0.01, gamma))