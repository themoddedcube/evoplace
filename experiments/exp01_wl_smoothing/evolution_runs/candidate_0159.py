import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-scale gamma schedule with a progress-based annealing
    floor and late-stage accuracy cap. High gamma while density is congested,
    smoothly decaying to accurate (low) gamma as the placement legalizes."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_ratio = math.log(gamma_high / gamma_low)

    # Primary driver: overflow on a log scale. Density is the physically
    # meaningful signal -- keep gamma high (smooth gradients) while bins are
    # congested, drop it as the layout approaches legalization.
    ov_term = ov ** 0.8

    # Guaranteed monotone decay (cosine) so gamma still anneals even if overflow
    # plateaus; prevents the schedule from getting stuck at high (inaccurate) gamma.
    prog_decay = 0.5 - 0.5 * math.cos(math.pi * progress)   # 0 -> 1
    prog_term = 1.0 - prog_decay                            # 1 -> 0

    # Mostly overflow-driven, with progress ensuring eventual fine-tuning.
    blend = 0.7 * ov_term + 0.3 * prog_term
    blend = min(1.0, max(0.0, blend))

    gamma = gamma_low * math.exp(log_ratio * blend)

    # Plateau detection: if recent HPWL has stagnated, nudge toward accurate
    # gamma to escape the smooth-but-wrong approximation.
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 6:
            improve = (recent[0] - min(recent[-3:])) / recent[0]
            if improve < 1e-3:
                gamma *= 0.85

    # Late-stage accuracy cap: once nearly converged, force low gamma so the
    # HPWL surrogate matches true wirelength for the final fine-tuning.
    if progress > 0.9 and ov < 0.10:
        gamma = min(gamma, 0.7)
    elif progress > 0.80 and ov < 0.15:
        gamma = min(gamma, 1.2)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))