import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress coupling and plateau-aware
    fine-tuning. High gamma while cells are spread (high overflow / early), low
    gamma for accurate HPWL once the layout settles."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: overflow. DREAMPlace-style smooth coupling where a still
    # spread-out layout (high overflow) keeps gradients smooth, and a settled
    # layout (low overflow) sharpens toward accurate HPWL.
    # Map overflow in [0,1] -> log-gamma in [log(gamma_low), log(gamma_high)].
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)
    ov_curve = ov ** 0.85  # slightly front-loaded so gamma drops as ov clears
    gamma_ov = math.exp(log_lo + (log_hi - log_lo) * ov_curve)

    # Secondary driver: iteration progress provides a monotone backstop so gamma
    # anneals even if the overflow signal is noisy or stalls.
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** progress

    # Blend: lean on overflow early, hand off to the progress backstop late.
    w_prog = progress ** 1.5
    gamma = (1.0 - w_prog) * gamma_ov + w_prog * gamma_prog

    # Plateau / divergence handling from HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-5:] if h is not None and h > 0]
        if len(recent) >= 3:
            best_recent = min(recent)
            prev = None
            if len(hpwl_history) >= 6 and hpwl_history[-6] and hpwl_history[-6] > 0:
                prev = hpwl_history[-6]
            else:
                prev = recent[0]

            # Stalled improvement: sharpen toward accurate HPWL for fine-tuning.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75

            # Rising / diverging HPWL: smooth gradients back out to recover.
            if recent[-1] > recent[0] * 1.02:
                gamma *= 1.4

    # Late-stage cap: guarantee accurate HPWL approximation near the end.
    if progress > 0.85:
        cap = 1.0 - 0.4 * (progress - 0.85) / 0.15  # 1.0 -> 0.6 over the tail
        gamma = min(gamma, cap)

    if not math.isfinite(gamma):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))