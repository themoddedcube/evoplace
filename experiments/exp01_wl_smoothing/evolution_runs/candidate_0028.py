import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma with exponential progress decay and HPWL-plateau sharpening."""

    # Guard against degenerate inputs.
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    # Smooth high->low schedule over the run: 8.0 early -> ~0.5 late.
    gamma_high = 8.0
    gamma_low = 0.5
    # Exponential (geometric) decay reads better than linear for WA-WL smoothing.
    base = gamma_high * (gamma_low / gamma_high) ** progress

    # Overflow-adaptive boost: while cells are still spread across many full bins,
    # keep gradients smooth; as overflow collapses, let gamma drop for accurate HPWL.
    # Maps overflow in [0,1] to a multiplicative factor in ~[0.6, 3.0].
    overflow_factor = 0.6 + 2.4 * (ov ** 1.5)
    gamma = base * overflow_factor

    # Late-stage plateau detection: if HPWL has stopped improving, sharpen (lower gamma)
    # to escape the smoothed approximation and refine true wirelength.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.7  # plateaued: push toward accurate regime

        # Diverging HPWL -> back off to smoother gradients for stability.
        if recent[-1] > recent[0] * 1.02:
            gamma *= 1.5

    # Floor gamma in the final fraction so we always fine-tune on accurate HPWL.
    if progress > 0.85:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))