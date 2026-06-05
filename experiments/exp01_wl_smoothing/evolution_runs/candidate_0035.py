import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress floor and gentle HPWL feedback."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Overflow is the primary physical signal: while cells are still
    # spread out (high overflow) keep gradients smooth; as the layout
    # settles (low overflow) sharpen toward an accurate HPWL surrogate.
    # DREAMPlace-style log-scaled coupling on overflow.
    ov_term = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov)

    # Progress provides a monotone backstop so gamma cannot stay high
    # if overflow plateaus without fully resolving.
    prog_term = gamma_high * (gamma_low / gamma_high) ** progress

    # Blend: early iterations trust overflow, late iterations trust
    # progress to guarantee fine-tuning even under stubborn overflow.
    w = progress
    gamma = (1.0 - w) * ov_term + w * prog_term

    # Gentle, bounded HPWL feedback. No multiplicative blow-up that
    # could destabilize the run (the cause of divergence in the prior
    # schedule); only mild nudges within a clamped band.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)
        # Stagnation: accuracy matters more -> ease gamma down a touch.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.9
        # Divergence: HPWL climbing -> add a little smoothing, capped.
        if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
            gamma = min(gamma * 1.2, gamma_high)

    # Smooth descending ceiling: never allow large gamma late in the run.
    ceiling = gamma_high * (gamma_low / gamma_high) ** progress * 1.5
    gamma = min(gamma, ceiling)

    # Hard fine-tuning cap in the final phase for an accurate HPWL.
    if progress > 0.9:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))