import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-decayed gamma schedule for differentiable placement."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Cosine-annealed geometric interpolation in log-space: smooth high->low decay
    # that lingers high early (cells still clustering) and eases into low late.
    cos_prog = 0.5 * (1.0 - math.cos(math.pi * progress))
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)
    base = math.exp(log_hi * (1.0 - cos_prog) + log_lo * cos_prog)

    # Overflow coupling: when the layout is still congested (high overflow) we
    # want smoother gradients, so push gamma up; as bins clear, relax toward base.
    # Blend so overflow dominates early and progress dominates late.
    overflow_factor = 0.55 + 2.0 * (ov ** 1.3)
    blend = 1.0 - 0.5 * progress  # trust overflow less near the end
    gamma = base * (1.0 + blend * (overflow_factor - 1.0))

    # Adaptive response to HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)
        finite = [h for h in recent if h == h and abs(h) != float("inf")]

        # Plateau detection -> sharpen approximation to push HPWL lower.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.65

        # Divergence / oscillation -> smooth gradients to recover stability.
        if len(finite) >= 2 and finite[0] > 0 and finite[-1] > finite[0] * 1.02:
            gamma *= 1.6

        # Mild oscillation damping based on variance of recent finite history.
        if len(finite) >= 3:
            mean = sum(finite) / len(finite)
            if mean > 0:
                var = sum((h - mean) ** 2 for h in finite) / len(finite)
                cv = math.sqrt(var) / mean
                if cv > 0.05:
                    gamma *= 1.0 + min(0.4, cv)

    # Final-phase clamp: late iterations need accurate HPWL, so cap gamma low,
    # but only once density is under control to avoid freezing a bad layout.
    if progress > 0.85:
        cap = 1.0 if ov < 0.15 else 2.0
        gamma = min(gamma, cap)
    elif progress > 0.65:
        gamma = min(gamma, 4.0)

    return min(50.0, max(0.01, gamma))