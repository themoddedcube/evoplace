import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # --- sanitize inputs (guard against NaN/inf/None -> avoids inf score) ---
    try:
        ov = float(overflow)
    except (TypeError, ValueError):
        ov = 1.0
    if not math.isfinite(ov):
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    T = total_iterations if (isinstance(total_iterations, int) and total_iterations > 0) else 1
    it = iteration if isinstance(iteration, int) and iteration >= 0 else 0
    progress = min(1.0, max(0.0, it / float(T)))

    g_hi, g_lo = 8.0, 0.5

    # --- primary signal: overflow-adaptive (placement-state driven) ---
    # log-linear interp in gamma between low-overflow (accurate) and high-overflow (smooth).
    # ov~0.1 -> ~g_lo, ov~1.0 -> ~g_hi
    ov_frac = min(1.0, max(0.0, (ov - 0.1) / 0.9))
    log_lo, log_hi = math.log(g_lo), math.log(g_hi)
    gamma_ov = math.exp(log_lo + ov_frac * (log_hi - log_lo))

    # --- secondary signal: cosine-annealed iteration schedule (fallback/smoothing) ---
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    gamma_it = math.exp(log_lo + cos * (log_hi - log_lo))

    # weight toward overflow early, lean on iteration schedule late for fine-tuning
    w = 0.7
    gamma = w * gamma_ov + (1.0 - w) * gamma_it

    # --- plateau detection: if HPWL stalls, drop gamma to sharpen approximation ---
    if isinstance(hpwl_history, (list, tuple)) and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-4:]
                  if isinstance(h, (int, float)) and math.isfinite(h)]
        if len(recent) >= 4 and recent[0] > 0:
            rel = abs(recent[-1] - recent[0]) / abs(recent[0])
            if rel < 1e-3:
                gamma *= 0.8

    if not math.isfinite(gamma):
        gamma = g_hi
    return min(50.0, max(0.01, gamma))