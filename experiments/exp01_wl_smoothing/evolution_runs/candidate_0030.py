import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Evolve gamma: high early for smooth gradients, low late for accurate HPWL,
    modulated by overflow and HPWL-progress feedback."""

    # --- robust normalization ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing: exponential decay blended with cosine ---
    # exponential gives geometric sweep across the wide dynamic range,
    # cosine softens the early plateau so cells stay clustered longer.
    expo = gamma_high * (gamma_low / gamma_high) ** progress
    cos = gamma_low + 0.5 * (gamma_high - gamma_low) * (1.0 + math.cos(math.pi * progress))
    base = 0.5 * expo + 0.5 * cos

    # --- overflow coupling ---
    # While cells are still spread (high overflow) keep gamma elevated for
    # smooth, long-range gradients; relax it as the layout legalizes.
    # Tie strength to progress so late iterations are not over-inflated.
    overflow_factor = 0.55 + 1.9 * (ov ** 1.4)
    overflow_factor = 1.0 + (overflow_factor - 1.0) * (1.0 - 0.5 * progress)
    gamma = base * overflow_factor

    # --- HPWL feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # stagnation -> sharpen approximation to escape flat region
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.7

            # divergence (HPWL climbing) -> smooth gradients back out
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

            # strong steady improvement -> let it keep refining (lower gamma)
            if prev > 0 and (prev - window[-1]) / prev > 1e-2:
                gamma *= 0.9

    # --- late-stage clamp for accurate final HPWL ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.2)

    if gamma != gamma:  # NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))