import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive cosine-annealed gamma schedule for WA-WL smoothing."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing: smooth log-cosine decay from high -> low ---
    # cos_prog goes 0 -> 1 with an ease-in/ease-out shape, giving long
    # high-gamma plateau early (cells still spreading) and gentle low-gamma tail.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # When density is still high we want smoother gradients (higher gamma).
    # Blend the time-based anneal with an overflow-driven target so that a
    # placement that is converging on density can drop gamma faster.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 0.85)

    # progress-weighted blend: trust the clock more early, density more late
    w_ov = 0.35 + 0.45 * progress
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # never let overflow alone keep gamma pinned high once we are deep in the run
    if progress > 0.5:
        gamma = min(gamma, base * 1.25)

    # --- HPWL-history feedback (robust to None / NaN / short history) ---
    if hpwl_history and len(hpwl_history) >= 4:
        clean = [h for h in hpwl_history[-8:]
                 if h is not None and h == h and h > 0.0 and h != float("inf")]
        if len(clean) >= 4:
            window = clean[-4:]
            first, last = window[0], window[-1]
            best = min(window)

            # plateau: relative improvement over the window is tiny -> sharpen
            if first > 0 and (first - best) / first < 1.5e-3:
                gamma *= 0.85

            # divergence: HPWL climbing -> back off, re-smooth gradients
            if last > first * 1.01:
                gamma *= 1.25

    # --- final-stage tightening for accurate HPWL ---
    if progress > 0.80:
        ceil = 1.5 if ov > 0.10 else 0.8
        gamma = min(gamma, ceil)
    if progress > 0.95:
        gamma = min(gamma, 0.5 if ov <= 0.10 else 1.0)

    # --- guard against NaN/Inf and clamp to legal range ---
    if not (gamma == gamma) or gamma in (float("inf"), float("-inf")):
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))