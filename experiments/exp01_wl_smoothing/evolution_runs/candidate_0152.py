import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-primary gamma schedule with progress floor and HPWL trend adaptation."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary signal: overflow is the true convergence indicator of placement.
    # Geometric interpolation: ov=1 -> gamma_high, ov=0 -> gamma_low.
    # Monotone and smooth in overflow, the proven DREAMPlace-style coupling.
    ov_term = gamma_low * (gamma_high / gamma_low) ** ov

    # Secondary signal: a cosine-annealed geometric progress floor, so gamma
    # keeps descending even if overflow temporarily stalls on a plateau.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_term = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Take the smaller: BOTH the spreading state (overflow) and the iteration
    # budget (progress) must permit a high gamma. Keeps it from staying smooth
    # too long, which is the main HPWL leak in the late phase.
    gamma = min(ov_term, prog_term)

    # HPWL trend adaptation: small, bounded nudges only.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Diverging HPWL -> approximation too loose; firm gamma up.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25
            # Steady improvement -> safe to refine; ease gamma down.
            elif window[-1] < window[0] * 0.99:
                gamma *= 0.93

            # Flat plateau -> sharpen the approximation to break the stall.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

    # End-game caps: enforce accurate HPWL once cells are spread out.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))