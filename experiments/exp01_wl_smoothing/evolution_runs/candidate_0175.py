import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware log-cosine gamma schedule with plateau/divergence control."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:               # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary annealing: geometric decay shaped by a cosine ease ---
    # cos_prog runs 0 -> 1, slow at the ends, fast in the middle, giving a
    # gentle start (keep cells clustered) and a gentle, accurate finish.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # When density is still high we want smoother gradients (higher gamma) so
    # cells keep spreading; as overflow drops we trust the sharper HPWL approx.
    # Blend a multiplicative term (relative) with an additive floor (absolute).
    ov_mult = 0.60 + 1.45 * (ov ** 1.2)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    gamma = 0.6 * base * ov_mult + 0.4 * ov_add

    # --- HPWL feedback: react to plateaus and divergence ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Plateau: barely improving -> sharpen to chase accuracy.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: HPWL climbing -> smooth out to recover stability.
            if first > 0:
                ratio = last / first
                if ratio > 1.02:
                    gamma *= min(1.45, 1.0 + 1.5 * (ratio - 1.0))
                elif ratio < 0.98:
                    gamma *= 0.93

    # --- late-stage ceilings: force an accurate finish once nearly placed ---
    if progress > 0.90:
        ceil = 1.2 if ov > 0.10 else 0.5
        gamma = min(gamma, ceil)
    elif progress > 0.80:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.0)
    elif progress > 0.65:
        gamma = min(gamma, 3.0 if ov > 0.10 else 1.8)

    # --- final guards ---
    if gamma != gamma:                     # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))