import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-anchored gamma schedule with progress decay and plateau control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Progress-driven core ---------------------------------------------
    # Cosine-eased log-interpolation: stays high through the spreading phase,
    # then decays steeply once cells have clustered. This keeps gradients
    # smooth early and sharpens HPWL accuracy late.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    decay_prog = 0.65 * cos_prog + 0.35 * (progress ** 1.5)
    base = gamma_high * (gamma_low / gamma_high) ** decay_prog

    # --- Overflow anchoring -----------------------------------------------
    # Overflow is the most physically grounded convergence signal: high
    # overflow => cells still overlapping => keep gamma high for smoothness;
    # low overflow => layout settled => drop gamma to recover true HPWL.
    ov_anchor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    ov_mult = 0.50 + 1.65 * (ov ** 1.2)

    # Blend: weight the overflow anchor more as the run progresses, since the
    # schedule clock is unreliable once the placement has effectively converged.
    w_ov = 0.35 + 0.30 * progress
    gamma = (1.0 - w_ov) * (base * ov_mult) + w_ov * ov_anchor

    # --- HPWL-history feedback --------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            # Plateau in best HPWL -> sharpen to chase finer wirelength.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging (HPWL climbing) -> back off, smooth gradients.
            if last > first * 1.02:
                gamma *= 1.40
            # Steady improvement -> nudge sharper to accelerate.
            elif last < first * 0.985:
                gamma *= 0.92

    # --- Late-stage ceilings ----------------------------------------------
    # Force low gamma near the end for HPWL fidelity, but allow more headroom
    # while overflow remains non-trivial to avoid re-spreading instabilities.
    if progress > 0.85:
        gamma = min(gamma, 1.4 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))