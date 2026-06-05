import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for differentiable global placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Primary anneal: geometric decay in log-space, eased by a cosine ramp.
    # Keeps gamma high while cells are still clustering, then glides down.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    log_base = math.log(gamma_high) + (math.log(gamma_low) - math.log(gamma_high)) * cos_prog
    base = math.exp(log_base)

    # --- Overflow coupling: overflow is the truest signal of "how spread out".
    # When overflow is high the placement is still legalizing, so favor smoother
    # (higher) gamma; as overflow collapses, trust the anneal and sharpen.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    # Blend the schedule-driven and overflow-driven targets. Early on we lean on
    # the anneal; late we lean on overflow so we don't sharpen a still-illegal layout.
    w_ov = 0.30 + 0.45 * progress
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # Gentle multiplicative nudge so extremes of overflow still register.
    gamma *= 0.70 + 0.55 * (ov ** 1.2)

    # --- HPWL feedback: react to plateaus and divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            rel_gain = (prev - best_recent) / prev if prev > 0 else 0.0

            if last > first * 1.02:
                # Diverging: HPWL climbing -> smooth gradients to recover.
                gamma *= 1.30
            elif rel_gain < 1e-3:
                # Plateaued: sharpen to squeeze out remaining wirelength.
                gamma *= 0.80
            elif last < first * 0.97:
                # Healthy descent: ease down slightly to refine.
                gamma *= 0.93

    # --- Endgame caps: never leave gamma high once we should be fine-tuning,
    # but allow a touch more smoothing if the layout is still over-dense.
    if progress > 0.85:
        gamma = min(gamma, 1.6 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.6 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))