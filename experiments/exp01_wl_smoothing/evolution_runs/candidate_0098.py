import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Base annealing: geometric (log-linear) decay shaped by a smooth
    #     cosine ramp so gamma stays high while cells are still clustering
    #     and falls off late for accurate HPWL.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow coupling. Placement quality tracks density spreading much
    #     more than raw iteration count, so let overflow dominate the target.
    #     High overflow -> keep gamma elevated (smooth gradients to keep
    #     spreading); low overflow -> push gamma down toward accurate regime.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)

    # Blend: schedule sets the envelope, overflow steers within it. Weight the
    # overflow signal up as the run progresses (early iters are noisy).
    w_ov = 0.35 + 0.35 * progress
    gamma = (1.0 - w_ov) * base + w_ov * (0.5 * base + 0.5 * ov_target)

    # --- HPWL feedback: react to convergence dynamics.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau in best HPWL -> sharpen (lower gamma) to refine.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging / oscillating up -> smooth out (raise gamma).
            if window[-1] > window[0] * 1.02:
                gamma *= 1.40
            # Healthy steady improvement -> nudge toward accuracy.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.92

    # --- Late-stage accuracy ceiling, relaxed only if density not yet legal.
    if progress > 0.88:
        ceil = 1.3 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.72:
        gamma = min(gamma, 2.4 if ov > 0.10 else 1.4)

    # Keep gamma high enough early to avoid premature freezing.
    if progress < 0.15 and ov > 0.20:
        gamma = max(gamma, 3.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))