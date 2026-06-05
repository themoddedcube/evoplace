import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL smoothing."""

    # --- robust input sanitation ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_ratio = math.log(gamma_low / gamma_high)  # negative

    # --- base annealing: geometric decay along a cosine-eased progress ---
    # cosine easing keeps gamma high a bit longer early, then drops smoothly.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * math.exp(log_ratio * cos_prog)

    # --- overflow coupling ---
    # The placement is only "ready" for low gamma once cells have spread out.
    # While overflow is high, hold gamma up regardless of iteration progress.
    # Blend a progress-driven anneal with an overflow-driven floor.
    overflow_target = gamma_low + (gamma_high - gamma_low) * (ov ** 0.75)

    # Weight toward overflow control early, toward the schedule late.
    w_sched = progress
    gamma = (1.0 - w_sched) * overflow_target + w_sched * base

    # Mild multiplicative nudge: very congested -> push smoother gradients.
    gamma *= 0.85 + 0.30 * (ov ** 1.5)

    # --- HPWL feedback (plateau / divergence detection) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # Plateau: progress has stalled -> sharpen approximation to refine.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence / oscillation: HPWL climbing -> smooth gradients back out.
            if window[0] > 0 and window[-1] > window[0] * 1.02:
                gamma *= 1.35

            # Steady, healthy descent late: lean into accuracy.
            if progress > 0.5 and window[-1] < window[0] * 0.995:
                gamma *= 0.92

    # --- late-stage accuracy ceiling ---
    # Once nearly converged, force low gamma for HPWL fidelity, but only when
    # density is acceptable; otherwise keep some smoothing to fix overflow.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        ceil = 2.5 if ov > 0.10 else 1.2
        gamma = min(gamma, ceil)

    return min(50.0, max(0.01, gamma))