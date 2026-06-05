import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma schedule with exponential decay and plateau control."""

    # --- robust input sanitation (avoid NaN/inf propagation) ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    it = iteration if (iteration is not None and iteration >= 0) else 0
    progress = it / total
    if progress != progress:          # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow
    if ov is None or ov != ov:        # None / NaN guard
        ov = 1.0
    ov = min(1.0, max(0.0, float(ov)))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base exponential decay over iteration progress ---
    base = gamma_high * (gamma_low / gamma_high) ** progress

    # --- overflow drives the real schedule: cells still clustered -> keep smooth ---
    # When overflow is high we want large gamma (smooth gradients); as the layout
    # legalizes (overflow -> 0) we trust the WL approximation and sharpen.
    overflow_factor = 0.5 + 2.5 * (ov ** 1.3)

    # blend iteration-decay with overflow-driven term (overflow weighted more)
    gamma = (0.35 * base + 0.65 * (gamma_high * (gamma_low / gamma_high) ** (1.0 - ov))) \
            * (0.4 + 0.6 * overflow_factor / 1.6)

    # gentle cosine smoothing of the iteration component to avoid abrupt jumps
    cos_mix = 0.5 * (1.0 + math.cos(math.pi * progress))
    gamma = gamma * (0.7 + 0.3 * cos_mix)

    # --- plateau / divergence detection from HPWL history ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            prev = recent[0]
            best_recent = min(recent[-5:])
            # stagnation: sharpen to push wirelength down
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.7
            # divergence (HPWL climbing): re-smooth to stabilize
            if recent[-1] > recent[0] * 1.02:
                gamma *= 1.4

    # --- late-stage sharpening cap for accurate final HPWL ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.5)

    if gamma != gamma or gamma in (float("inf"), float("-inf")):
        gamma = gamma_low

    return min(50.0, max(0.01, float(gamma)))