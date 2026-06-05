import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with smooth progress decay.

    High gamma while cells are still clustered (high overflow / early
    progress) for smooth, well-conditioned gradients; low gamma late for an
    accurate HPWL approximation during fine-tuning. Overflow is the primary
    driver (it reflects the *actual* spreading state), with progress as a
    monotone backstop and a gentle plateau nudge. Deliberately avoids the
    large gamma "spikes" of the previous version, which can destabilize the
    coupled placement/density optimization and diverge HPWL.
    """

    # --- sanitize inputs --------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_high = math.log(gamma_high)
    log_low = math.log(gamma_low)

    # --- overflow-driven component (primary) ------------------------------
    # DREAMPlace-style: gamma tracks how spread the placement actually is.
    # ov ~ 1.0 (clustered)  -> gamma near gamma_high
    # ov ~ 0.0 (spread out) -> gamma near gamma_low
    # Smoothstep in overflow keeps the transition gentle and monotone.
    s = ov * ov * (3.0 - 2.0 * ov)                # smoothstep(ov)
    g_ov = math.exp(log_low + (log_high - log_low) * s)

    # --- progress-driven component (backstop) -----------------------------
    # Monotone geometric (cosine-eased) decay so gamma cannot stay pinned
    # high if overflow stalls; guarantees fine-tuning regime near the end.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    g_prog = math.exp(log_high + (log_low - log_high) * cos_prog)

    # Blend: lean on overflow early, on progress late (ensures convergence).
    w = progress                                  # 0 -> overflow, 1 -> progress
    gamma = math.exp((1.0 - w) * math.log(g_ov) + w * math.log(g_prog))

    # --- gentle plateau / divergence response -----------------------------
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 6:
            window = recent[-5:]
            prev = recent[-6]
            best_recent = min(window)

            # Stagnation: sharpen (lower gamma) to chase a better HPWL,
            # but only mildly so the optimizer stays stable.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.90

            # HPWL climbing: too sharp/noisy -> smooth a little.
            # Bounded, modest bump (no large spikes that cause oscillation).
            if window[-1] > window[0] * 1.03:
                gamma *= 1.15
            elif window[-1] < window[0] * 0.97:
                gamma *= 0.96

    # --- late-stage ceiling: force accurate-HPWL regime -------------------
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.08 else 0.6)
    elif progress > 0.75:
        gamma = min(gamma, 2.0 if ov > 0.08 else 1.0)

    # --- finalize ---------------------------------------------------------
    if gamma != gamma:                            # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))