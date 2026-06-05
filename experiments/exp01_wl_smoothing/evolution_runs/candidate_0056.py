import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma annealing for differentiable global placement.

    High gamma early (smooth gradients while cells are clustered / overflow
    is high) decaying to low gamma late (accurate HPWL for fine-tuning).
    Overflow is the primary physical driver; progress provides a stable
    fallback so the schedule never stalls. Plateau / divergence handling is
    gentle and bounded to avoid the instability that sends HPWL to inf.
    """

    # --- normalize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- progress-based geometric decay (monotone, well-behaved baseline) ---
    base = gamma_high * (gamma_low / gamma_high) ** progress

    # --- overflow coupling (DREAMPlace-style): gamma tracks remaining ---
    #     density spread. Smooth, bounded multiplier in roughly [0.55, 2.5].
    overflow_factor = 0.55 + 1.95 * (ov ** 1.2)

    # Blend the two drivers rather than multiplying raw (multiplying can
    # over-amplify and destabilize gradients). Weight overflow more early,
    # progress more late, so the tail reliably anneals toward gamma_low.
    w_ov = 1.0 - progress
    gamma = base * (w_ov * overflow_factor + (1.0 - w_ov) * 1.0)

    # --- gentle plateau / divergence response (bounded) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)

        # Stuck: improvement negligible -> lower gamma slightly for sharper
        # (more accurate) gradients to escape the plateau.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.85

        # Diverging: recent HPWL climbing -> smooth gradients a bit, but
        # cap the boost so we never blow up.
        if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
            gamma *= 1.25

    # --- smooth late-stage annealing (no hard cliff) ---
    if progress > 0.8:
        tail = (progress - 0.8) / 0.2          # 0 -> 1 over last 20%
        ceiling = 2.0 * (1.0 - tail) + 0.5 * tail
        gamma = min(gamma, ceiling)

    return min(50.0, max(0.01, gamma))