import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma schedule with exponential decay and plateau control."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary schedule: smooth cosine-eased exponential decay ---
    # high gamma early (smooth gradients, cluster cells) -> low gamma late
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # When density is still spread out (high overflow), keep gamma higher to
    # maintain smooth gradients; when bins are settling, sharpen toward HPWL.
    overflow_target = gamma_low + (gamma_high - gamma_low) * (ov ** 0.9)

    # blend the time-based decay with the overflow-driven target, weighting the
    # overflow signal more in the middle of the run where it is most informative
    w_ov = 0.4 + 0.2 * math.sin(math.pi * progress)
    gamma = (1.0 - w_ov) * base + w_ov * overflow_target

    # --- HPWL feedback: detect plateau / divergence ---
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-7:] if (h is not None and h == h and h > 0)]
        if len(recent) >= 4:
            window = recent[-4:]
            best_recent = min(window)
            ref = recent[-5] if len(recent) >= 5 else window[0]

            # plateau: little improvement -> reduce gamma to refine HPWL
            if ref > 0 and (ref - best_recent) / ref < 1e-3:
                gamma *= 0.88

            # divergence: HPWL climbing -> raise gamma to re-smooth gradients
            if window[-1] > window[0] * 1.015:
                gamma *= 1.25

    # --- late-stage ceiling: force fine-tuning once placement is legal-ish ---
    if progress > 0.85:
        ceil = 1.2 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)

    return min(50.0, max(0.01, gamma))