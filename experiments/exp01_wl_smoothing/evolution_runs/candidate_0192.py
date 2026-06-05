import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma annealing with anti-divergence guardrails."""

    gamma_high = 8.0
    gamma_low = 0.5

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    # --- primary signal: overflow-driven (DREAMPlace-style log interpolation) ---
    # overflow ~1.0 (cells overlapping) -> high gamma (smooth gradients)
    # overflow ~0.1 (cells spread)      -> low gamma  (accurate HPWL)
    t_ov = (min(1.0, max(0.10, ov)) - 0.10) / 0.90
    t_ov = min(1.0, max(0.0, t_ov))
    gamma_ov = gamma_low * (gamma_high / gamma_low) ** t_ov

    # --- secondary signal: cosine anneal on iteration (guarantees cool-down) ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_iter = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # trust overflow more (it tracks physical settling); iteration ensures progress
    gamma = 0.65 * gamma_ov + 0.35 * gamma_iter

    # --- gentle HPWL-history feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # HPWL climbing -> re-smooth (bounded, prevents oscillation)
            if window[-1] > window[0] * 1.01:
                gamma = min(gamma_high, gamma * 1.25)
            # stalled -> sharpen gently
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.90

    # --- late-stage refinement ceilings ---
    if progress > 0.90 and ov < 0.10:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 3.0 if ov > 0.15 else 1.5)

    # --- anti-divergence floor: never collapse gamma while heavily overlapping ---
    if ov > 0.5:
        gamma = max(gamma, 2.0)
    elif ov > 0.25:
        gamma = max(gamma, 1.0)

    # --- final NaN / range guard ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))