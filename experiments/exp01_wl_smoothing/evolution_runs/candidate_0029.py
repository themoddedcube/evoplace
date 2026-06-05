import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven WA-WL smoothing schedule with progress floor and plateau adaptation."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: overflow. DREAMPlace ties smoothness directly to
    # density convergence, not iteration count. Map overflow exponentially
    # so gamma falls fast once cells start to spread (ov drops below ~0.7).
    # ov=1.0 -> ~8.0, ov=0.5 -> ~2.3, ov=0.1 -> ~0.78, ov=0 -> 0.5.
    overflow_gamma = gamma_low * (gamma_high / gamma_low) ** (ov ** 0.85)

    # Progress provides a monotone descending envelope so that even if
    # overflow plateaus high, we still anneal toward accurate HPWL late.
    progress_cap = gamma_high * (gamma_low / gamma_high) ** progress

    # Use the smaller of the two: never smoother than either the
    # overflow-implied need or the schedule envelope allows.
    gamma = min(overflow_gamma, progress_cap)

    # Keep enough smoothness early to let cells cluster (avoid noisy
    # gradients before any structure forms).
    if progress < 0.1:
        gamma = max(gamma, 4.0)

    # HPWL trend adaptation.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)

        # Stagnation: relative improvement over the window is tiny.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            # If density is essentially resolved, sharpen to chase HPWL;
            # otherwise smooth slightly to escape a gradient stall.
            if ov < 0.15:
                gamma *= 0.85
            else:
                gamma *= 1.15

        # Divergence: HPWL climbing -> gradients too noisy, smooth more.
        if recent[-1] > recent[0] * 1.02:
            gamma *= 1.4

    # Final fine-tuning phase: force accurate approximation.
    if progress > 0.85:
        gamma = min(gamma, 0.9)

    return min(50.0, max(0.01, gamma))