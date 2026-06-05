import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-coupled cosine-annealed geometric gamma schedule.

    High gamma early (smooth gradients, cells cluster) decaying to low gamma
    late (accurate HPWL, fine placement), with overflow gating so the decay
    only commits once density actually relaxes, plus plateau/divergence
    feedback from the HPWL trace.
    """

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Base trajectory: geometric (log-linear) decay along a cosine path.
    # Cosine easing keeps gamma high a touch longer early, then anneals fast
    # through the middle and flattens near the end for stable fine-tuning.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow coupling. Overflow is the true placement state, not the
    # iteration counter: when bins are still saturated we must keep gradients
    # smooth regardless of how far along we are; once overflow drops we let the
    # accurate low-gamma regime take over. Blend a multiplicative term (scales
    # the trajectory) with an additive floor (guarantees smoothing under
    # heavy congestion). The blend tilts toward the additive term while
    # overflow is high and toward the trajectory as it clears.
    ov_mult = 0.50 + 1.65 * (ov ** 1.20)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.40)
    w_add = 0.30 + 0.30 * ov          # 0.30 (clear) .. 0.60 (saturated)
    gamma = (1.0 - w_add) * base * ov_mult + w_add * ov_add

    # --- HPWL feedback: react to what the optimizer is actually doing.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            # Stagnation: best HPWL barely improving over the window -> sharpen
            # the approximation to extract finer wirelength gains.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: HPWL climbing -> back off to smoother gradients to
            # let the placement re-settle. Scale the response to the severity.
            if first > 0 and last > first * 1.005:
                climb = min(0.5, (last - first) / first)
                gamma *= 1.0 + 0.8 * (climb / 0.5)
            # Steady descent: trust it, nudge toward sharper accuracy.
            elif first > 0 and last < first * 0.98:
                gamma *= 0.93

    # --- Late-stage accuracy ceiling. Near the end, only allow high gamma if
    # density is still genuinely problematic; otherwise force the accurate
    # low-gamma regime so reported HPWL reflects the true placement.
    if progress > 0.85:
        ceil = 1.4 if ov > 0.10 else 0.65
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.4 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))