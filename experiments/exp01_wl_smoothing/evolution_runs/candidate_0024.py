import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # --- sanitize inputs ---
    if total_iterations is None or total_iterations <= 0:
        total_iterations = 1
    it = iteration if iteration is not None else 0
    if it < 0:
        it = 0
    if it > total_iterations:
        it = total_iterations
    progress = it / float(total_iterations)          # 0 -> 1 over the run

    try:
        ovf = float(overflow)
    except (TypeError, ValueError):
        ovf = 1.0
    if math.isnan(ovf) or math.isinf(ovf):
        ovf = 1.0
    ovf = min(1.0, max(0.0, ovf))

    # --- base overflow-adaptive term (DREAMPlace-style log sweep) ---
    # high overflow (cells spread/clustered) -> large gamma (smooth)
    # low overflow (nearly legal)            -> small gamma (accurate HPWL)
    base = 10.0 ** ((ovf - 0.1) * (20.0 / 9.0) - 1.0)   # ~0.4 .. ~10 for ovf in [0.1,1]
    base = 4.0 * base

    # --- iteration schedule: exponential decay from high to low gamma ---
    gamma_hi = 8.0
    gamma_lo = 0.5
    decay = gamma_hi * (gamma_lo / gamma_hi) ** progress   # 8.0 -> 0.5 geometrically

    # --- cosine-annealed blend: trust overflow early, force fine-tuning late ---
    w = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 early -> 0 late
    gamma = w * base + (1.0 - w) * decay

    # --- plateau detection: if HPWL has stalled, sharpen (lower gamma) ---
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-6:]
                  if h is not None and not math.isnan(h) and not math.isinf(h)]
        if len(recent) >= 4:
            prev = sum(recent[:len(recent) // 2]) / (len(recent) // 2)
            curr = sum(recent[len(recent) // 2:]) / (len(recent) - len(recent) // 2)
            if prev > 0 and abs(prev - curr) / prev < 1e-3:
                gamma *= 0.7   # nudge toward accuracy when progress stalls

    # --- final safety clamp ---
    if math.isnan(gamma) or math.isinf(gamma):
        gamma = 1.0
    return min(50.0, max(0.01, gamma))