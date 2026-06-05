import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-annealed gamma schedule for WA-WL placement."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing: blend cosine (smooth) with geometric (fast tail) ---
    # cosine gives a gentle high-gamma plateau early, sharp drop mid, soft landing late
    cos_t = 0.5 * (1.0 + math.cos(math.pi * progress))          # 1 -> 0
    cos_base = gamma_low + (gamma_high - gamma_low) * cos_t
    geo_base = gamma_high * (gamma_low / gamma_high) ** progress
    base = 0.5 * (cos_base + geo_base)

    # --- overflow adaptivity ---
    # While cells are still spread out (high overflow) keep gamma higher for
    # smooth gradients; as the placement legalizes (low overflow) trust the
    # annealed base and let gamma fall toward gamma_low for an accurate HPWL.
    # Bounded, mean-1-ish multiplier so it nudges rather than explodes.
    overflow_factor = 0.75 + 0.75 * (ov ** 1.25)                # ~0.75 .. 1.5
    gamma = base * overflow_factor

    # --- HPWL feedback: react to stagnation / divergence ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # stagnating: sharpen the approximation to escape the plateau
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75

            # diverging (HPWL climbing): re-smooth gradients
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

    # --- late-phase cap: force fine-tuning regime near the end ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.5)

    # --- final guard ---
    if not (gamma == gamma):   # NaN safety
        gamma = gamma_low
    return min(50.0, max(0.01, float(gamma)))