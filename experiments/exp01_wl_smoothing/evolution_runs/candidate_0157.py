import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-coupled log-decay gamma schedule with plateau adaptation."""

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
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # --- base schedule: smooth log-space cosine decay (high -> low) ---
    # cos_prog goes 0 -> 1, easing slowly at both ends for a gentle landing.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = math.exp(log_hi + (log_lo - log_hi) * cos_prog)

    # --- overflow coupling ---
    # Placement physically governs accuracy: while bins are congested the HPWL
    # approximation must stay smooth, so blend the time-based curve toward an
    # overflow-driven floor. As overflow drains, the schedule is freed to
    # descend toward gamma_low regardless of iteration count.
    ov_curve = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    # weight on overflow term shrinks late so the time schedule dominates the
    # fine-tuning phase even if a few bins remain marginally over-dense.
    w_ov = (1.0 - progress) * 0.5 + 0.15
    gamma = (1.0 - w_ov) * base + w_ov * ov_curve

    # --- HPWL-history adaptation (gentle, bounded) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # plateau: best HPWL barely improving -> sharpen to fine-tune
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # divergence: HPWL climbing -> smooth gradients to recover
            if first > 0 and last > first * 1.02:
                gamma *= 1.30
            # steady improvement -> let it keep sharpening slightly
            elif first > 0 and last < first * 0.98:
                gamma *= 0.93

    # --- end-game ceilings: force accurate HPWL once placement is legal ---
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final guards ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))