import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with cosine backbone and plateau adaptation."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress < 0.0:
        progress = 0.0
    elif progress > 1.0:
        progress = 1.0

    try:
        ov = float(overflow)
    except (TypeError, ValueError):
        ov = 1.0
    if math.isnan(ov) or math.isinf(ov):
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- backbone: cosine annealing in log-space for a smooth high->low sweep ---
    # cosine gives a gentle start (cells still clustering) and a soft landing.
    cos_t = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    log_hi, log_lo = math.log(gamma_high), math.log(gamma_low)
    base = math.exp(log_lo + (log_hi - log_lo) * cos_t)

    # --- overflow coupling ---
    # When overflow is high the placement is still spread out, so we want
    # smoother gradients (higher gamma). As overflow collapses toward 0 we
    # trust HPWL accuracy and pull gamma down. Blend with the schedule backbone
    # so neither term alone dominates.
    overflow_factor = 0.55 + 1.85 * (ov ** 1.3)
    gamma = base * overflow_factor

    # --- plateau / divergence adaptation from HPWL history ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-5:]
                  if isinstance(h, (int, float)) and not math.isnan(h) and not math.isinf(h)]
        if len(recent) >= 2:
            prev = recent[0]
            best_recent = min(recent)

            # stalled improvement -> sharpen (lower gamma) to refine wirelength
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75

            # diverging (HPWL climbing) -> smooth (raise gamma) to recover
            if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
                gamma *= 1.4

    # --- late-stage cap: force accurate HPWL near the end ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    if math.isnan(gamma) or math.isinf(gamma):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))