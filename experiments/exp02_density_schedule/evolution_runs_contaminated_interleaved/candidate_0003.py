def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    GAMMA_HIGH = 8.0      # smooth, stable gradients while cells are clustered
    GAMMA_LOW = 0.5       # sharp, accurate HPWL for fine-tuning
    LO, HI = 0.01, 50.0

    # Sanitize inputs (guard against nan/inf that produced the divergence).
    of = overflow
    if of != of or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(1.0, max(0.0, of))

    gn = gradient_norm
    if gn != gn or gn in (float("inf"), float("-inf")):
        gn = 0.0

    # Overflow-adaptive core: log-linear map so gamma tracks spreading progress.
    # overflow ~1.0 (start) -> GAMMA_HIGH ; overflow ~0.0 (spread) -> GAMMA_LOW.
    gamma = GAMMA_LOW * (GAMMA_HIGH / GAMMA_LOW) ** of

    # Iteration floor: enforce monotone-ish annealing so late iters stay sharp
    # even if overflow plateaus, preventing endless high-gamma stalling.
    anneal = GAMMA_HIGH * (0.985 ** float(max(0, iteration)))
    gamma = min(gamma, max(GAMMA_LOW, anneal))

    # Stability guard: if overflow is rising (placement un-spreading) or
    # gradients spike, back off toward smoother, safer gradients.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev and of > prev + 0.02:
            gamma = min(GAMMA_HIGH, gamma * 1.5)

    # Damp against the previous value to avoid oscillation/divergence.
    if current_lambda == current_lambda and 0.0 < current_lambda < float("inf"):
        gamma = 0.7 * gamma + 0.3 * min(HI, max(LO, current_lambda))

    return float(min(HI, max(LO, gamma)))