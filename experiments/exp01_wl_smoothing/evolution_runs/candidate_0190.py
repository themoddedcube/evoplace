import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Base annealing: log-space cosine decay (smooth, monotone) ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow gating ---
    # Placement quality is governed far more by spreading state (overflow)
    # than by raw iteration count. While cells are still piled up (high ov)
    # we must keep gradients smooth; once spread (low ov) we sharpen toward
    # the true HPWL. Blend a multiplicative and an additive overflow term so
    # neither dominates at the extremes.
    ov_mult = 0.55 + 1.6 * (ov ** 1.25)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = 0.55 * base * ov_mult + 0.45 * ov_add

    # When density is essentially resolved, commit hard to accurate HPWL
    # regardless of where we are in the iteration budget.
    if ov < 0.06:
        gamma = min(gamma, gamma_low + 4.0 * ov)

    # --- HPWL trend feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Plateau: relative best-improvement stalling -> sharpen to escape
            # the over-smoothed minimum and recover true wirelength.
            if prev > 0:
                rel_gain = (prev - best_recent) / prev
                if rel_gain < 1e-4:
                    gamma *= 0.72
                elif rel_gain < 1e-3:
                    gamma *= 0.85

            # Divergence: HPWL climbing -> back off to smoother gradients,
            # but only meaningfully when density is not yet settled.
            if first > 0:
                ratio = last / first
                if ratio > 1.02:
                    gamma *= 1.30 if ov > 0.08 else 1.10
                elif ratio < 0.98:
                    gamma *= 0.93

    # --- Late-stage ceilings: force accurate HPWL in the endgame ---
    if progress > 0.90:
        ceil = 1.2 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.80:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.1)
    elif progress > 0.65:
        gamma = min(gamma, 3.0 if ov > 0.10 else 1.8)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))