import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-gamma schedule with progress annealing and
    stagnation control. Returns gamma in [0.01, 50.0]."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Core: overflow is the most reliable physical signal (DREAMPlace-style).
    # Map overflow -> gamma in log-space so the transition is smooth and the
    # dynamic range is fully exercised. High overflow (cells spread/clustered)
    # -> high gamma (smooth gradients); low overflow (legalizable) -> low gamma.
    ov_log = gamma_low * (gamma_high / gamma_low) ** ov

    # --- Progress backbone: cosine-in-log annealing as a safety floor so gamma
    # decays even if the overflow signal is sticky or noisy early on.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_log = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Weight overflow more as placement matures: early, trust the schedule;
    # late, trust the measured density so we can sharpen HPWL aggressively.
    w_ov = 0.45 + 0.35 * progress
    gamma = math.exp((1.0 - w_ov) * math.log(prog_log) + w_ov * math.log(ov_log))

    # --- HPWL feedback: adapt to convergence behavior.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            rel_gain = (prev - best_recent) / prev if prev > 0 else 0.0
            trend = (last - first) / first if first > 0 else 0.0

            # Diverging: HPWL climbing -> back off, re-smooth gradients.
            if trend > 0.015:
                gamma *= 1.30
            # Stagnating with little recent improvement -> sharpen to escape.
            elif rel_gain < 1e-3:
                gamma *= 0.80
            # Healthy descent -> nudge sharper to fine-tune.
            elif trend < -0.01:
                gamma *= 0.92

    # --- End-game caps: force accurate HPWL once density is acceptable.
    if progress > 0.88:
        ceil = 1.2 if ov > 0.10 else 0.55
        gamma = min(gamma, ceil)
    elif progress > 0.72:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.2)

    # Keep some smoothness while overflow is still high, regardless of phase.
    if ov > 0.85:
        gamma = max(gamma, 4.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))