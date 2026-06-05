import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-coupled gamma schedule with progress annealing and
    history-aware stabilization. High gamma while cells are clustered
    (high overflow / early), low gamma for accurate HPWL fine-tuning."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                       # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- 1) Overflow-driven term (physically primary in DREAMPlace) ---
    # Cells still overlapping -> need smooth (high) gamma; spread out -> sharpen.
    # Smooth-step on overflow so mid-range overflow doesn't whipsaw gamma.
    ov_s = ov * ov * (3.0 - 2.0 * ov)              # smoothstep(ov)
    gamma_ov = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_s)

    # --- 2) Progress-driven log-cosine anneal (monotone backbone) ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- 3) Blend: lean on overflow early, on the clock late ---
    # Late in the run overflow is small but we still want a firm push to low gamma.
    w_prog = 0.35 + 0.45 * progress                # 0.35 -> 0.80
    gamma = (1.0 - w_prog) * gamma_ov + w_prog * gamma_prog

    # --- 4) History-aware adjustment ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Plateau: best barely improving -> sharpen to escape the flat region.
            if prev > 0.0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> back off, smooth the landscape.
            if last > first * 1.02:
                gamma *= 1.30
            # Healthy descent -> nudge sharper for tighter wirelength.
            elif last < first * 0.98:
                gamma *= 0.92

    # --- 5) Late-stage ceilings (protect HPWL accuracy near convergence) ---
    if progress > 0.85:
        gamma = min(gamma, 1.4 if ov > 0.10 else 0.65)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    # --- final clamp ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))