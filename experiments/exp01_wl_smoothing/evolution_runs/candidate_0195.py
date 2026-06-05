import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Smooth-to-sharp gamma schedule for WA-WL placement.

    High gamma early (smooth gradients, cells cluster), low gamma late
    (accurate HPWL, fine-tuning). Overflow-adaptive with gentle, bounded
    corrections so the optimizer stays stable.
    """

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:            # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base schedule: geometric (log-linear) decay on a cosine clock ---
    # cos_prog eases slowly at the start (keep cells smooth/clustered) and
    # accelerates the sharpening in the back half.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # When density is still spread out (high overflow) we want a smoother
    # objective; once bins are mostly legal (low overflow) we let gamma drop.
    # Blend the time-based schedule with a pure-overflow target. Keep the
    # multiplier near 1.0 to avoid runaway gamma.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.3)
    blend = 0.35 + 0.45 * progress          # trust the clock more over time
    gamma = (1.0 - blend) * ov_target + blend * base

    # mild multiplicative nudge from overflow, tightly bounded
    ov_mult = 0.85 + 0.30 * ov
    gamma *= ov_mult

    # --- HPWL feedback: gentle, bounded, no large spikes ---
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0 and h != float('inf')]
        if len(recent) >= 6:
            window = recent[-5:]
            prev = recent[-6]
            best_recent = min(window)

            # plateau: nudge gamma down to sharpen and break the stall
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.90

            # divergence (HPWL climbing): smooth gradients a touch, but
            # cap the correction so we never blow up the objective
            if window[0] > 0 and window[-1] > window[0] * 1.01:
                gamma *= 1.15
            # steady improvement: let gamma keep easing down
            elif window[0] > 0 and window[-1] < window[0] * 0.99:
                gamma *= 0.97

    # --- late-stage ceilings: force accurate HPWL near the end ---
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.08 else 0.6)
    elif progress > 0.75:
        gamma = min(gamma, 2.0 if ov > 0.08 else 1.2)
    elif progress > 0.55:
        gamma = min(gamma, 3.5)

    # --- final clamp / NaN guard ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))