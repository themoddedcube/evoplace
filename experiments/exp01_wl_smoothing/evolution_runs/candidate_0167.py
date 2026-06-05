import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress floor and plateau control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Primary driver: overflow ---
    # DREAMPlace-style behaviour: gamma tracks how spread-out the cells still are.
    # While bins are congested (high overflow) we want smooth gradients (high gamma);
    # as the layout legalizes (overflow -> 0) we sharpen toward an accurate HPWL.
    # Geometric interpolation in log-space keeps the transition smooth and avoids
    # collapsing to the floor too early.
    ov_term = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov)

    # --- Secondary driver: progress (cosine in log-space) ---
    # Guarantees monotone annealing even if overflow plateaus, so late iterations
    # always get a chance to fine-tune at low gamma.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_term = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: lean on overflow early, on the progress schedule late.
    w_prog = progress
    gamma = (1.0 - w_prog) * ov_term + w_prog * prog_term

    # Progress-dependent floor: never let early iterations drop too sharp
    # (noisy gradients early destabilize the spread), and allow deep sharpening late.
    early_floor = gamma_low + (gamma_high - gamma_low) * (1.0 - progress) * ov
    gamma = max(gamma, early_floor)

    # --- HPWL feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # Stalled improvement -> sharpen to escape the WA-WL approximation bias.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging (HPWL climbing) -> smooth back out to restabilize.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # --- Late-stage ceilings: commit to accurate HPWL once mostly legal ---
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))