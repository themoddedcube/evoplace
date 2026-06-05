import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Anneal gamma from coarse (smooth gradients) to fine (accurate HPWL),
    coupled to density overflow so smoothing relaxes only as cells spread out."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth log-space anneal as the primary driver (monotone, well-conditioned).
    base = gamma_high * (gamma_low / gamma_high) ** progress

    # Overflow keeps smoothing high while cells are still clustered, and lets it
    # fall toward gamma_low as the layout legalizes. Bounded in [0.7, 1.9].
    overflow_factor = 0.7 + 1.2 * (ov ** 1.2)
    gamma = base * overflow_factor

    # Gentle, bounded reactions to the optimization trajectory.
    if hpwl_history and len(hpwl_history) >= 6:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6]
        best_recent = min(recent)

        # Plateau: improvement stalled -> sharpen approximation to refine.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.85

        # Divergence: HPWL climbing -> re-smooth to recover stability.
        if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
            gamma *= 1.25

    # Late-phase fine-tuning: cap smoothing so HPWL stays accurate near the end.
    if progress > 0.85:
        cap = 1.0 - 0.5 * (progress - 0.85) / 0.15  # 1.0 -> 0.5
        gamma = min(gamma, max(gamma_low, cap))

    if not math.isfinite(gamma):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))