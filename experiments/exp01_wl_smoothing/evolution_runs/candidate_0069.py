import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealing of the WA-WL smoothing parameter gamma."""

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    if not (progress == progress):          # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base schedule: geometric (exponential) decay --------------------
    # Smooth gradients early (cells still spreading), accurate HPWL late.
    base = gamma_high * (gamma_low / gamma_high) ** progress

    # --- overflow coupling ----------------------------------------------
    # While density overflow is high the layout is still legalizing, so keep
    # gamma elevated for stable gradients; as overflow drains, let it relax.
    # Blend a progress-driven floor with an overflow-driven boost so that a
    # low-overflow region late in the run is allowed to sharpen aggressively.
    overflow_boost = 0.5 + 2.0 * (ov ** 1.2)
    gamma = base * overflow_boost

    # --- history-driven adaptation --------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: stalled improvement -> sharpen to chase finer HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75

            # Divergence: HPWL climbing -> smooth out to recover stability.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

    # --- late-stage cap: commit to accurate HPWL near the end -----------
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    # --- final clamp / NaN backstop -------------------------------------
    if not (gamma == gamma):
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))