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
    it = iteration if (iteration is not None and iteration == iteration) else 0
    progress = it / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- core schedule: geometric (log-linear) decay on a cosine-eased clock ---
    # Cosine easing keeps gamma high a bit longer early (cells still spreading)
    # then descends smoothly; geometric interpolation gives perceptually even
    # multiplicative steps between gamma_high and gamma_low.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # When the layout is still congested (high overflow) we want smoother
    # gradients => larger gamma; once overflow collapses we trust the sharper
    # (more accurate) HPWL approximation. Blend a multiplicative congestion
    # factor with an absolute overflow-driven floor/target.
    overflow_factor = 0.55 + 1.7 * (ov ** 1.3)
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.1)
    gamma = 0.55 * base * overflow_factor + 0.45 * ov_target

    # --- HPWL feedback: adapt to plateau / divergence ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: HPWL barely improving -> sharpen (lower gamma) to
            # refine wirelength, but only once cells are reasonably settled.
            if prev > 0 and (prev - best_recent) / prev < 1e-3 and ov < 0.25:
                gamma *= 0.82

            # Divergence: HPWL climbing -> back off to smoother gradients.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            elif window[-1] > window[0] * 1.005:
                gamma *= 1.12

    # --- late-stage clamp: force accurate HPWL near convergence ---
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:  # NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))