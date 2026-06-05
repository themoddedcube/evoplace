import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven cosine-decayed gamma with plateau adaptation."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Time-based backbone: cosine-annealed geometric decay from high -> low.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    time_base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow-based backbone: gamma should track the *physical* state of the
    # layout, not just the clock. While cells are still spread out (high ov),
    # keep gamma large for smooth gradients; as the layout legalizes (ov -> 0),
    # collapse toward gamma_low for an accurate HPWL approximation.
    ov_base = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)

    # Blend: early on trust the clock; later trust the measured overflow, since
    # the true placement state is what should gate the final sharpening.
    w_ov = 0.35 + 0.45 * progress
    gamma = (1.0 - w_ov) * time_base + w_ov * ov_base

    # Mild multiplicative overflow gate so a stubbornly high overflow keeps
    # gamma from dropping prematurely even late in the run.
    gamma *= 0.7 + 0.6 * (ov ** 1.1)

    # HPWL-history feedback: react to convergence dynamics.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: HPWL barely improving -> sharpen to escape the flat basin.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging: HPWL climbing -> smooth gradients to stabilize.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.35
            # Healthy descent: nudge gamma down to refine.
            elif window[-1] < window[0] * 0.97:
                gamma *= 0.93

    # Late-stage caps to guarantee an accurate final HPWL, relaxed only if the
    # layout is still illegal (high overflow).
    if progress > 0.88:
        gamma = min(gamma, 1.4 if ov > 0.10 else 0.6)
    elif progress > 0.72:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))