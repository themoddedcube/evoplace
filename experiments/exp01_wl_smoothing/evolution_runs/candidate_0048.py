import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma with iteration-based annealing and plateau control."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = min(1.0, max(0.0, iteration / total))
    ov = overflow if (overflow is not None and overflow == overflow) else 1.0  # NaN guard
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary signal: overflow-adaptive (DREAMPlace style) ---
    # high overflow (cells spread/clustered) -> smooth gradients (high gamma)
    # low overflow (legal-ish layout)         -> accurate HPWL (low gamma)
    # exponential map keeps the response well-conditioned across the range.
    ov_gamma = gamma_low * (gamma_high / gamma_low) ** ov

    # --- secondary signal: iteration annealing as a decaying ceiling ---
    # guarantees gamma trends downward even if overflow stalls high.
    anneal_ceiling = gamma_high * (gamma_low / gamma_high) ** progress

    # blend: take the more conservative (smaller) of the two so late
    # iterations cannot stay smooth if either signal says "refine now".
    gamma = min(ov_gamma, max(anneal_ceiling, gamma_low))

    # --- plateau / divergence handling from HPWL history ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h > 0]
        if len(recent) >= 2:
            prev = recent[0]
            best_recent = min(recent[1:])
            improvement = (prev - best_recent) / prev if prev > 0 else 0.0

            # stalled -> sharpen the approximation to chase real HPWL
            if improvement < 1e-3:
                gamma *= 0.75

            # diverging (HPWL climbing) -> smooth gradients to recover
            if recent[-1] > recent[0] * 1.02:
                gamma *= 1.4

    # --- final fine-tuning: force accurate HPWL near the end ---
    if progress > 0.85:
        gamma = min(gamma, 1.0)
    if progress > 0.95:
        gamma = min(gamma, 0.5)

    return min(50.0, max(0.01, gamma))