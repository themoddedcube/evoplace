import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware exponential gamma annealing for WA-WL placement."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base schedule: geometric (log-linear) decay in time ---
    # Smooth gradients early (cells still spreading), accurate HPWL late.
    base = gamma_high * (gamma_low / gamma_high) ** progress

    # --- overflow-driven schedule ---
    # While density overflow is high, cells are still clustered/legalizing,
    # so keep gamma high; as overflow drains, sharpen toward gamma_low.
    # ov in [0,1] -> gamma in [gamma_low, gamma_high], with emphasis on
    # staying smooth until overflow is genuinely low.
    ov_shaped = ov ** 0.5
    overflow_gamma = gamma_low + (gamma_high - gamma_low) * ov_shaped

    # --- blend: lean on overflow early, on the time schedule late ---
    # Early iterations trust the density signal; late iterations commit to
    # the annealing floor for fine HPWL tuning.
    w_time = progress
    gamma = (1.0 - w_time) * overflow_gamma + w_time * base

    # Never let the time term alone push gamma below what overflow warrants
    # by more than a little: if the placement is still very spread, keep it
    # somewhat smooth to avoid gradient noise stalling the spread.
    if ov > 0.85:
        gamma = max(gamma, 2.0)

    # --- HPWL-history feedback (plateau / divergence control) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0 and h != float("inf")]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: improvement has nearly stalled -> sharpen to chase HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> smooth gradients to recover.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25

    # --- late-stage ceiling: force accurate HPWL near convergence ---
    if progress > 0.80:
        ceil = 1.5 if ov > 0.10 else 0.8
        gamma = min(gamma, ceil)

    if gamma != gamma or gamma == float("inf"):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))