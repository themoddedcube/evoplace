import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for differentiable global placement."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:               # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base geometric anneal on a cosine-warped clock ---
    # cos warp keeps gamma high a bit longer early, then decays fast late.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # While bins are congested (high ov) we keep gradients smooth (raise gamma);
    # once cells spread out (low ov) we let gamma drop for accurate HPWL.
    # Blend a multiplicative and an additive overflow term for stability.
    ov_mult = 0.55 + 1.50 * (ov ** 1.25)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = 0.55 * base * ov_mult + 0.45 * ov_add

    # --- adaptive feedback from HPWL trajectory ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            w0, wlast = window[0], window[-1]

            # plateau: barely improving -> sharpen (lower gamma) to refine HPWL
            if prev > 0.0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # diverging: HPWL climbing -> smooth gradients (raise gamma) to recover
            if w0 > 0.0 and wlast > w0 * 1.02:
                gamma *= 1.30
            # healthy descent -> nudge sharper to keep tightening
            elif w0 > 0.0 and wlast < w0 * 0.98:
                gamma *= 0.93

    # --- late-stage ceilings: force accuracy near the end, but stay smoother
    #     while density is still unresolved (ov high) to avoid HPWL blow-ups ---
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.80:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.0)
    elif progress > 0.65:
        gamma = min(gamma, 3.0 if ov > 0.10 else 1.8)

    # --- final NaN/range guard ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))