import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing: smooth (high) -> accurate (low) ---
    # Cosine-shaped log-interpolation gives a gentle start, fast middle decay,
    # and a soft landing near the end.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # The cells are still clustered while overflow is high, so we want smoother
    # gradients (higher gamma); once overflow drops the layout is settling and we
    # can sharpen the HPWL approximation. Blend a progress-driven term with an
    # overflow-driven term so neither dominates.
    ov_term = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = 0.6 * base + 0.4 * ov_term

    # Mild multiplicative nudge from overflow so very-full layouts stay smooth.
    gamma *= 0.7 + 0.5 * ov

    # --- HPWL-history feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            last, first = window[-1], window[0]

            # Plateau: improvement has stalled -> sharpen to chase real HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> smooth gradients to restabilize.
            if last > first * 1.02:
                gamma *= 1.30
            # Healthy descent -> gently sharpen.
            elif last < first * 0.98:
                gamma *= 0.93

    # --- end-of-run accuracy caps ---
    # Late iterations must trust the true HPWL; clamp gamma low unless density
    # is still poor (overflow high), in which case keep some smoothing.
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.75:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.3)

    # --- final guards ---
    if gamma != gamma:  # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))