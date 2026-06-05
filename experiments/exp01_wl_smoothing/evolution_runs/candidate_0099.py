import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware geometric gamma anneal with plateau/divergence control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Base geometric (log-linear) anneal driven by a cosine ease.
    # Geometric interpolation keeps gamma changing multiplicatively, which
    # matches the multiplicative nature of the WA-WL smoothing scale.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow blending.
    # When cells are still spread out (high overflow) we want a smoother
    # (higher gamma) landscape; once density settles we trust the sharper
    # approximation. Blend a multiplicative term with an additive floor so
    # gamma never collapses while overflow is still large.
    ov_mult = 0.60 + 1.40 * (ov ** 1.20)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.50)
    gamma = 0.60 * base * ov_mult + 0.40 * ov_add

    # --- HPWL feedback: react to stagnation and divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Plateau: relative improvement tiny -> sharpen to refine HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: HPWL climbing -> smooth out to recover stability.
            if first > 0 and last > first * 1.02:
                gamma *= 1.40
            # Healthy descent: keep sharpening gently.
            elif first > 0 and last < first * 0.98:
                gamma *= 0.92

    # --- Late-phase ceilings: force fine-tuning regime near the end, but
    # relax the cap if overflow is still meaningfully high (legality first).
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))