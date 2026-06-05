def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    GAMMA_HI = 8.0
    GAMMA_LO = 0.5

    # Overflow-driven target: high gamma while cells are still spread out
    # (overflow high), low gamma once density has settled (overflow low).
    ov = overflow if overflow is not None else 1.0
    ov = min(max(ov, 0.0), 1.0)

    # Map overflow -> gamma target on a log scale (smooth, monotone).
    # ov ~ 1.0  -> GAMMA_HI ; ov -> 0.0 -> GAMMA_LO
    base = GAMMA_LO * ((GAMMA_HI / GAMMA_LO) ** ov)

    # Iteration prior: gentle exponential floor so we keep annealing even if
    # overflow plateaus early (avoids getting stuck at high gamma).
    decay = max(0.97 ** float(iteration), 0.02)
    iter_target = GAMMA_LO + (GAMMA_HI - GAMMA_LO) * decay

    # Blend overflow signal with iteration prior (overflow leads).
    target = 0.7 * base + 0.3 * iter_target

    # Plateau detection: if overflow has stalled, push gamma down harder
    # to escape the stagnant region and sharpen the HPWL approximation.
    if overflow_history is not None and len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        spread = max(recent) - min(recent)
        if spread < 1e-3:
            target *= 0.85

    # Gradient guard: if gradients explode, raise gamma a touch for smoother
    # descent; if they vanish, allow gamma to drop for finer tuning.
    if gradient_norm is not None and gradient_norm > 0.0:
        if gradient_norm > 5.0:
            target *= 1.10
        elif gradient_norm < 0.05:
            target *= 0.95

    # Smooth toward target from current value to avoid oscillation.
    cur = current_lambda if current_lambda is not None else target
    alpha = 0.5
    new_lambda = (1.0 - alpha) * cur + alpha * target

    # Never let gamma increase late in the run.
    if new_lambda > cur and iteration > 20:
        new_lambda = cur

    return float(min(max(new_lambda, 0.01), 50.0))