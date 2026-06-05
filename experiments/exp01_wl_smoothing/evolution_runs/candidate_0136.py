import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma annealing for differentiable global placement.

    Backbone: gamma tracks overflow geometrically (clustered cells -> smooth
    high gamma; spread cells -> accurate low gamma). A progress floor guarantees
    fine-tuning late even if overflow stalls. HPWL feedback nudges around that.
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
    log_ratio = math.log(gamma_high / gamma_low)  # > 0

    # --- primary signal: geometric in overflow ---------------------------
    # ov ~ 1.0 -> gamma_high ; ov ~ 0.0 -> gamma_low.
    # A mild exponent (>1) keeps gamma high while overflow is still large
    # and drops it sharply only once the placement is genuinely spread out.
    ov_shaped = ov ** 1.3
    gamma_ov = gamma_low * math.exp(log_ratio * ov_shaped)

    # --- secondary signal: cosine annealing on progress ------------------
    # Guarantees monotone-ish smoothing -> accuracy transition even if the
    # overflow signal is noisy or plateaus.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)   # 0 -> 1, smooth ends
    gamma_prog = gamma_low * math.exp(log_ratio * (1.0 - cos_prog))

    # Blend: lean on overflow early (it is the most informative density cue),
    # hand control to the progress schedule late (when overflow is small and
    # we must commit to accurate gradients).
    w_prog = progress
    gamma = (1.0 - w_prog) * gamma_ov + w_prog * gamma_prog

    # --- HPWL feedback: gentle, bounded multipliers ----------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # Plateau: best HPWL barely improving -> sharpen toward accuracy.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging: HPWL climbing -> smooth gradients to recover.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            # Healthy descent -> nudge slightly sharper.
            elif window[-1] < window[0] * 0.97:
                gamma *= 0.93

    # --- late-stage accuracy ceiling -------------------------------------
    # Force low gamma at the end so the final HPWL approximation is faithful,
    # but relax the ceiling while density overflow is still meaningful.
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.75:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.2)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))