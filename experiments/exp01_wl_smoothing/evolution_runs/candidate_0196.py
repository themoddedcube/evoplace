import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-annealed gamma for WA-WL placement.

    Primary signal is overflow (the physical convergence proxy): gamma is
    interpolated in log-space from gamma_high (cells spread, overflow~1) down
    to gamma_low (cells settled, overflow~0). A cosine progress term guarantees
    monotone cooling even if overflow stalls. HPWL feedback nudges gently and
    a late-stage ceiling protects final HPWL accuracy.
    """

    # ---- robust inputs ----
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    ratio = gamma_low / gamma_high                # < 1

    # ---- overflow-driven log interpolation (DREAMPlace-style) ----
    # ov=1 -> gamma_high, ov=0 -> gamma_low. Slight convexity (ov**1.1) keeps
    # gamma high a touch longer while density is still being resolved.
    ov_key = ov ** 1.1
    gamma_ov = gamma_high * (ratio ** (1.0 - ov_key))

    # ---- progress-driven cosine cooling (fail-safe monotone descent) ----
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)   # 0 -> 1, smooth ends
    gamma_prog = gamma_high * (ratio ** cos_prog)

    # ---- blend in log-space (geometric mean): smooth, divergence-resistant ----
    w_ov = 0.62
    gamma = (gamma_ov ** w_ov) * (gamma_prog ** (1.0 - w_ov))

    # ---- gentle HPWL feedback ----
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Diverging WL -> smooth gradients to restabilize (modest).
            if last > first * 1.02:
                gamma *= 1.18
            # Plateaued improvement -> sharpen accuracy (lower gamma).
            elif prev > 0.0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.88
            # Steady improvement -> let it keep cooling slightly.
            elif last < first * 0.98:
                gamma *= 0.96

    # ---- late-stage accuracy ceilings (prevent blow-up, sharpen final HPWL) ----
    if progress > 0.85:
        gamma = min(gamma, 1.4 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    # ---- final clamp ----
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))