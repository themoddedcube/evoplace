import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma with progress-based annealing."""
    # Normalized progress in [0, 1]
    T = max(1, total_iterations)
    p = min(1.0, max(0.0, iteration / T))

    # Clamp overflow to a sane range
    ov = min(1.0, max(0.0, overflow))

    # --- Base annealing from progress (high -> low) ---
    # Cosine annealing between a high and low gamma gives smooth gradients
    # early (cells still clustering) and accurate HPWL late (fine-tuning).
    g_hi, g_lo = 8.0, 0.5
    base = g_lo + 0.5 * (g_hi - g_lo) * (1.0 + math.cos(math.pi * p))

    # --- Overflow adaptation ---
    # When overflow is still high, density is far from satisfied, so keep
    # gamma high (smoother) regardless of iteration. As overflow drops we
    # trust the progress-based schedule and push gamma low for accuracy.
    # DREAMPlace-style exponential mapping of overflow to a multiplier.
    ov_gamma = g_lo * 10.0 ** (2.0 * ov)  # 0.5 .. 50 as ov goes 0 -> 1

    # Take the smoother (larger) of the two while overflow is meaningful,
    # so we never anneal faster than density allows.
    gamma = max(base, ov_gamma * min(1.0, ov / 0.1))

    # --- Plateau detection: if HPWL has stalled, sharpen approximation ---
    if len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        finite = [h for h in recent if h == h and abs(h) != float("inf")]
        if len(finite) >= 4:
            lo, hi = min(finite), max(finite)
            denom = abs(hi) + 1e-12
            if (hi - lo) / denom < 1e-3:
                # Stalled: lower gamma to refine the wirelength estimate.
                gamma *= 0.7

    # --- Late-stage hard push toward accuracy ---
    if p > 0.85 and ov < 0.1:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))