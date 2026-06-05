import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # --- sanitize inputs (guards against the inf/NaN failure mode) ---
    if total_iterations is None or total_iterations <= 0:
        total_iterations = 1
    if iteration is None or iteration < 0:
        iteration = 0
    progress = min(1.0, max(0.0, float(iteration) / float(total_iterations)))

    if overflow is None or math.isnan(overflow) or math.isinf(overflow):
        overflow = 1.0
    overflow = min(1.0, max(0.0, float(overflow)))

    # --- overflow-adaptive core (DREAMPlace-style log mapping) ---------
    # High overflow (cells still clustered) -> large gamma (smooth grads).
    # Low overflow (legalizable layout)     -> small gamma (accurate HPWL).
    # Maps overflow in [0.1, 1.0] -> gamma in ~[0.5, 8.0] on a log scale.
    of = max(overflow, 0.05)
    gamma_overflow = 8.0 * 10.0 ** ((of - 1.0) * (math.log10(8.0 / 0.5) / 0.9))

    # --- iteration-based cosine annealing floor -----------------------
    # Guarantees monotone-ish cooling even if overflow plateaus/stalls,
    # so late iterations always get fine-tuning gradients.
    g_hi, g_lo = 8.0, 0.5
    gamma_cosine = g_lo + 0.5 * (g_hi - g_lo) * (1.0 + math.cos(math.pi * progress))

    # --- blend: overflow leads early, schedule enforces late cooldown -
    w = progress  # trust the iteration floor more as we approach the end
    gamma = (1.0 - w) * gamma_overflow + w * min(gamma_overflow, gamma_cosine)

    # --- stagnation detection: if HPWL stops improving, sharpen -------
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-4:]
                  if h is not None and not math.isnan(h) and not math.isinf(h)]
        if len(recent) >= 2:
            denom = abs(recent[0]) + 1e-12
            rel_impr = (recent[0] - recent[-1]) / denom
            if rel_impr < 1e-4:          # plateaued -> cool faster
                gamma *= 0.85

    # --- final clamp ---------------------------------------------------
    if math.isnan(gamma) or math.isinf(gamma):
        gamma = 1.0
    return min(50.0, max(0.01, float(gamma)))