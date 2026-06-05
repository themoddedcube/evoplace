def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Overflow-adaptive WA-WL smoothing schedule.
    # High gamma early (cells overlap, high overflow -> smooth gradients),
    # low gamma late (layout settled, low overflow -> accurate HPWL).
    GAMMA_MIN, GAMMA_MAX = 0.5, 8.0

    # Defensive input clamping (NaN/out-of-range guards).
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)
    it = max(int(iteration), 0)

    # DREAMPlace-style log-scale map: overflow~1 -> GAMMA_MAX, overflow~0 -> GAMMA_MIN.
    gamma = GAMMA_MIN * (GAMMA_MAX / GAMMA_MIN) ** of

    # Iteration annealing floor: forces sharpening even if overflow stalls
    # (guards against reward-hacking placements that keep overflow pinned high).
    decay = 0.999 ** float(it)
    gamma = min(gamma, GAMMA_MAX * max(decay, GAMMA_MIN / GAMMA_MAX))

    # Trend damping: smooth more when overflow rises, sharpen when converging.
    if isinstance(overflow_history, list) and len(overflow_history) >= 2:
        a, b = overflow_history[-1], overflow_history[-2]
        if a == a and b == b:
            gamma *= 1.05 if (a - b) > 0 else 0.97

    # Gradient-norm stability guard.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            gamma *= 1.10
        elif gradient_norm < 1e-3:
            gamma *= 0.90

    return float(min(max(gamma, 0.01), 50.0))