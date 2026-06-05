import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-coupled annealing schedule for WA-WL gamma."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Geometric (log-linear) base anneal, eased so gamma stays high
    #     while cells are still spreading and drops fast near the end. ---
    ease = progress * progress * (3.0 - 2.0 * progress)   # smoothstep
    base = gamma_high * (gamma_low / gamma_high) ** ease

    # --- Overflow coupling (DREAMPlace-style): density congestion is the
    #     real signal for how much smoothing the gradient still needs.
    #     overflow ~1 -> keep gamma near high; overflow ~0 -> let base win. ---
    ov_factor = ov ** 1.3
    overflow_target = gamma_low + (gamma_high - gamma_low) * ov_factor

    # Blend base anneal with the overflow target; lean on overflow early
    # (when it is most informative) and on the time-base late.
    w_ov = 0.65 * (1.0 - ease) + 0.20
    gamma = w_ov * overflow_target + (1.0 - w_ov) * base

    # --- HPWL feedback: adapt only with enough clean history. ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Diverging: HPWL climbing -> gradients too noisy, smooth more.
            if last > first * 1.01:
                gamma *= 1.30
            # Plateau with no improvement -> push toward accurate regime.
            elif prev > 0 and (prev - best_recent) / prev < 5e-4:
                gamma *= 0.80
            # Healthy descent -> ease gamma down gently for accuracy.
            elif last < first * 0.99:
                gamma *= 0.93

    # --- Late-stage accuracy ceiling: HPWL approximation must be tight
    #     at the end, but stay relaxed if density is still unresolved. ---
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.08 else 0.6)
    elif progress > 0.75:
        gamma = min(gamma, 2.2 if ov > 0.08 else 1.3)
    elif progress > 0.55:
        gamma = min(gamma, 4.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))