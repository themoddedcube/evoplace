import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware log-cosine gamma annealing for differentiable placement."""

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    if math.isinf(ov):
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing: cosine in log-space ----------------------------
    # Smooth high->low decay; cosine gives slower descent early (keep cells
    # clustered) and a gentle tail (stable fine-tuning), unlike pure exp.
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)
    cos_frac = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    base = math.exp(log_lo + (log_hi - log_lo) * cos_frac)

    # --- overflow coupling ----------------------------------------------
    # Placement is driven primarily by density legalization. When overflow is
    # still high, favour smoother (higher) gamma; as bins clear, let gamma
    # drop toward the accurate regime. Bounded multiplier in [0.55, 2.5].
    overflow_factor = 0.55 + 1.95 * (ov ** 1.2)
    gamma = base * overflow_factor

    # --- HPWL-history feedback ------------------------------------------
    clean = [h for h in hpwl_history if isinstance(h, (int, float))
             and h == h and not math.isinf(h)] if hpwl_history else []
    if len(clean) >= 5:
        recent = clean[-5:]
        prev = clean[-6] if len(clean) >= 6 else recent[0]
        best_recent = min(recent)

        # Stagnation: HPWL barely improving -> sharpen approximation a bit
        # so the optimizer chases true wirelength instead of the smooth proxy.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.75

        # Divergence: HPWL climbing -> smooth the landscape to recover.
        if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
            gamma *= 1.4

    # --- end-game clamp: commit to accurate regime ----------------------
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.5)

    if not (gamma == gamma) or math.isinf(gamma):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))