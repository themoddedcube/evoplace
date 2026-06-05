import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive cosine gamma schedule for differentiable global placement."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base decay: cosine-annealed log-interpolation ---
    # cosine warps progress so gamma stays high while cells are still clustering
    # (early plateau) and drops smoothly toward the end for accurate HPWL.
    warped = 0.5 - 0.5 * math.cos(math.pi * progress)
    log_hi, log_lo = math.log(gamma_high), math.log(gamma_low)
    base = math.exp(log_hi + (log_lo - log_hi) * warped)

    # --- overflow coupling ---
    # Density is the physical signal for "how spread out are the cells".
    # High overflow (cells piled up) -> keep gamma high to smooth gradients.
    # Low overflow (legal-ish layout) -> let gamma fall for fine HPWL tuning.
    # Bounded multiplier in [0.7, 2.2] to avoid destabilizing blow-ups.
    overflow_factor = 0.7 + 1.5 * (ov ** 1.25)
    gamma = base * overflow_factor

    # --- HPWL trend feedback (guarded against NaN/inf/non-positive) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:]
                  if isinstance(h, (int, float)) and math.isfinite(h) and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            # plateau: improvement stalled -> sharpen (lower gamma) for precision
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.8
            # divergence: HPWL climbing -> re-smooth (raise gamma) to recover
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

    # --- late-stage cap: ensure accurate approximation near convergence ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    # --- final clamp + NaN guard ---
    if not math.isfinite(gamma):
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))