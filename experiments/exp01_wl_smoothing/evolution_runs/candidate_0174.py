import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware geometric gamma annealing for differentiable global placement.

    High gamma early (smooth gradients while cells are clustered / overflow high),
    annealed geometrically to low gamma late (accurate HPWL for fine-tuning).
    Overflow gates the decay so gamma stays high until density actually spreads,
    which prevents premature low-gamma noise that can diverge the placement.
    """

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_ratio = math.log(gamma_low / gamma_high)  # negative

    # --- effective decay coordinate -------------------------------------
    # Blend schedule progress with overflow so the anneal is driven by the
    # *physical* state of the placement, not just the iteration counter.
    # When overflow is still high we hold the coordinate back (keep gamma high);
    # when bins have cleared we let it advance toward the low-gamma regime.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)      # smooth 0->1
    ov_block = ov ** 0.7                                     # 1 when full, ->0 when clear
    # decay coordinate t in [0,1]: advances with progress, held back by overflow
    t = cos_prog * (1.0 - 0.45 * ov_block) + 0.10 * (1.0 - ov_block)
    t = min(1.0, max(0.0, t))

    # geometric (log-linear) interpolation between gamma_high and gamma_low
    gamma = gamma_high * math.exp(log_ratio * t)

    # --- mild overflow floor while density is high ----------------------
    # Don't let any adjustment push gamma too low while cells are still
    # heavily overlapping; noisy gradients there are the main divergence risk.
    if ov > 0.15:
        floor = gamma_low + (gamma_high - gamma_low) * min(1.0, ov ** 1.3) * 0.35
        gamma = max(gamma, floor)

    # --- HPWL-history feedback (gentle, bounded) ------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # diverging HPWL -> gradients too noisy -> smooth them out
            if first > 0 and last > first * 1.015:
                gamma *= 1.25
            # steady improvement -> sharpen the approximation a bit
            elif first > 0 and last < first * 0.99:
                gamma *= 0.93
            # plateau (no meaningful improvement) -> nudge toward accuracy
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.88

    # --- late-stage accuracy caps ---------------------------------------
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.8)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))