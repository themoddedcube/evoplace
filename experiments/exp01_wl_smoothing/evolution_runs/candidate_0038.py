import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-led, progress-annealed gamma schedule for WA-WL placement."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = min(1.0, max(0.0, iteration / total))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_high = math.log(gamma_high)
    log_low = math.log(gamma_low)

    # Primary driver: overflow interpolates gamma in log-space.
    # High overflow (cells spread/clustered) -> smooth high gamma;
    # low overflow (placement settling) -> accurate low gamma.
    # sqrt keeps gamma elevated while overflow is still moderate.
    ov_term = ov ** 0.5
    gamma_ov = math.exp(log_low + (log_high - log_low) * ov_term)

    # Safety floor: pure progress exponential decay guarantees annealing
    # even if overflow plateaus high. Take the gentler of the two but never
    # let overflow alone keep gamma far above the decay envelope.
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** progress
    gamma = min(gamma_ov, gamma_prog * 1.5)
    gamma = max(gamma, gamma_low * 0.5)

    # Adaptive response to the HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 6:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6]
        best_recent = min(recent)
        # Plateau: sharpen gradients to escape (mild, bounded).
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.85
        # Rising HPWL (instability): smooth more, but capped to avoid blow-up.
        if recent[-1] > recent[0] * 1.02:
            gamma = min(gamma * 1.3, gamma_high)

    # Final fine-tuning phase: force accurate, low gamma.
    if progress > 0.9:
        gamma = min(gamma, 0.8)

    return min(50.0, max(0.01, gamma))