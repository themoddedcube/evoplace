import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Anneal gamma from smooth->accurate, modulated by overflow and HPWL trend."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    if ov != ov:  # NaN guard
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Cosine-shaped geometric anneal: holds gamma high while cells cluster,
    # then drops smoothly for accurate fine-tuning near the end.
    shaped = 0.5 * (1.0 - math.cos(math.pi * progress))  # ease-in-out in [0,1]
    base = gamma_high * (gamma_low / gamma_high) ** shaped

    # Overflow keeps gamma elevated while density is unresolved; once spread out,
    # let gamma fall so HPWL approximation tightens.
    overflow_factor = 0.55 + 2.0 * (ov ** 1.3)
    gamma = base * overflow_factor

    # HPWL-trend adaptation.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-5:] if h is not None and h == h and h > 0]
        if len(recent) >= 3:
            prev = None
            if len(hpwl_history) >= 6 and hpwl_history[-6] and hpwl_history[-6] > 0:
                prev = hpwl_history[-6]
            else:
                prev = recent[0]
            best_recent = min(recent)

            # Plateau: sharpen toward accurate HPWL to escape a flat region.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.65

            # Divergence: HPWL climbing -> smooth gradients to restabilize.
            if recent[-1] > recent[0] * 1.02:
                gamma *= 1.4

    # Cap gamma in the final phase so the placement converges on accurate HPWL.
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.2)

    if gamma != gamma:  # final NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))