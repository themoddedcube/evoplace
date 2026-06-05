import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress-based annealing floor.

    Keeps gamma high while the placement is still spread out (high overflow /
    early progress) for smooth gradients, then anneals toward an accurate,
    low-gamma regime as the layout legalizes. A light HPWL-history term nudges
    gamma down on plateaus and up on divergence.
    """

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- overflow term ---------------------------------------------------
    # Overflow is the true physical state of the layout. Map it
    # geometrically between the low and high gamma bounds so that a full
    # layout (ov=1) -> gamma_high and a legalized layout (ov=0) -> gamma_low.
    # Use a mild concave exponent so gamma stays high until overflow has
    # genuinely dropped, then falls off for fine-tuning.
    ov_shaped = ov ** 0.85
    gamma_ov = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_shaped)

    # --- progress term ---------------------------------------------------
    # Cosine annealing in log-space provides a smooth, schedule-only decay
    # that does not depend on overflow. This guards against cases where the
    # overflow signal is missing, stuck, or noisy.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: trust overflow more early, schedule more late.
    w = progress
    gamma = (1.0 - w) * gamma_ov + w * 0.5 * (gamma_ov + gamma_prog)

    # --- HPWL-history feedback ------------------------------------------
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0.0]
        if len(recent) >= 4:
            window = recent[-4:]
            first, last = window[0], window[-1]
            best = min(window)

            # Plateau: relative improvement over the window is tiny -> sharpen
            # the approximation (lower gamma) to refine HPWL.
            if first > 0.0 and (first - best) / first < 1.0e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> smooth out gradients (raise gamma).
            if last > first * 1.02:
                gamma *= 1.25

    # --- late-stage ceiling ---------------------------------------------
    # Once we are deep into the run, force accuracy unless the layout is
    # still meaningfully overflowed.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)

    return min(50.0, max(0.01, gamma))