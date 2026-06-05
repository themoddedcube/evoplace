import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress coupling and
    plateau-aware sharpening for differentiable global placement.

    Strategy:
      - Primary control is overflow: while bins are congested the cells
        still need to spread, so keep gamma high (smooth gradients).
        As overflow collapses, drop gamma to sharpen the WA-WL
        approximation and recover accurate HPWL.
      - Progress provides a monotone floor/ceiling envelope so the
        schedule still anneals even if the overflow signal is noisy or
        stalls, and guarantees a low-gamma fine-tuning tail.
      - HPWL history nudges gamma: sharpen on plateau, stay smooth on
        divergence.
    """

    # ---- sanitize inputs ----
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:            # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # ---- overflow-driven base (log-spaced, DREAMPlace-style) ----
    # ov in [0,1] -> gamma in [gamma_low, gamma_high] geometrically.
    # Convex exponent keeps gamma high until overflow is genuinely low,
    # then drops it quickly for the fine-tuning regime.
    ov_drive = ov ** 1.35
    base_ov = gamma_low * (gamma_high / gamma_low) ** ov_drive

    # ---- progress-driven cosine anneal (smooth high -> low) ----
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # ---- blend: lean on overflow early, progress takes over late ----
    w_prog = progress ** 0.85
    gamma = (1.0 - w_prog) * base_ov + w_prog * base_prog

    # ---- HPWL-history feedback ----
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # plateau: improvement nearly exhausted -> sharpen to chase HPWL
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # divergence: HPWL climbing -> smooth gradients to recover
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            elif window[-1] < window[0] * 0.985:
                gamma *= 0.95

    # ---- monotone envelope from progress (anneal even if overflow noisy) ----
    # ceiling decays with progress; floor guarantees gradients don't vanish.
    ceil_env = gamma_high * (gamma_low / gamma_high) ** (progress ** 1.2)
    gamma = min(gamma, max(ceil_env, gamma_low))

    # ---- low-gamma fine-tuning tail ----
    if progress > 0.85:
        gamma = min(gamma, 1.4 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    # ---- final guards ----
    if gamma != gamma:                  # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))