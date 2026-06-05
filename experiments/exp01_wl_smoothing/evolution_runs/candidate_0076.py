import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware gamma annealing for differentiable global placement."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth (cosine) geometric interpolation in log-space between high and low.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    sched = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow gate: while cells are still badly spread, keep gradients smooth.
    # Blend the time-schedule with an overflow-driven target. When overflow is
    # high we trust the schedule less and push gamma up; when overflow has
    # collapsed we let the low-gamma fine-tuning regime dominate.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 0.7)
    blend = ov ** 0.5  # weight toward overflow target when overflow is large
    gamma = (1.0 - blend) * sched + blend * ov_target

    # Trust the schedule's lower bound less early on: never anneal faster than
    # overflow allows, so we don't sharpen gradients before cells legalize.
    floor_by_overflow = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = max(gamma, min(sched, floor_by_overflow) if ov > 0.05 else gamma_low)

    # HPWL-history feedback: react to stagnation and divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stagnation: relative improvement nearly zero -> sharpen to refine.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: recent HPWL climbing -> re-smooth gradients.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.35

            # Mild oscillation damping around the trend.
            mean_w = sum(window) / len(window)
            if mean_w > 0:
                spread = (max(window) - min(window)) / mean_w
                if spread > 0.05:
                    gamma *= 1.10

    # Late-phase ceiling: once we are deep into placement, cap gamma so the
    # final HPWL approximation stays accurate, but relax the cap if overflow
    # is still meaningful (cells not yet legal).
    if progress > 0.80:
        ceil = 2.0 if ov > 0.10 else (0.7 if progress > 0.92 else 1.0)
        gamma = min(gamma, ceil)

    return min(50.0, max(0.01, gamma))