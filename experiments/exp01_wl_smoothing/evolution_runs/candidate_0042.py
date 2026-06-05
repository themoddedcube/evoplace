import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Evolve gamma: high early for smooth clustering, low late for accurate HPWL.

    Combines geometric (exponential) decay with a cosine-shaped easing,
    modulated by density overflow and HPWL stagnation/divergence signals.
    """

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base decay: blend geometric decay with cosine easing ---
    # geometric decay gives smooth multiplicative falloff
    geo = gamma_high * (gamma_low / gamma_high) ** progress
    # cosine annealing holds gamma higher early, drops sharply mid-late
    cos = gamma_low + 0.5 * (gamma_high - gamma_low) * (1.0 + math.cos(math.pi * progress))
    # weight toward cosine early (keeps cells clustered), geometric late
    w = progress
    base = (1.0 - w) * cos + w * geo

    # --- overflow adaptation ---
    # When density overflow is high, cells still overlap: keep gamma elevated
    # for smoother gradients. As overflow clears, let gamma fall to its base.
    overflow_factor = 0.55 + 1.65 * (ov ** 1.3)
    gamma = base * overflow_factor

    # --- HPWL-history feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)

        # stagnation: best HPWL barely improving -> sharpen approximation
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.8

        # divergence: HPWL climbing -> smooth gradients to recover stability
        if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
            gamma *= 1.4

    # --- late-stage cap for accurate final wirelength ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 2.0)

    return min(50.0, max(0.01, gamma))