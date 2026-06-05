def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    GAMMA_HIGH = 8.0
    GAMMA_LOW = 0.5
    E = 2.718281828459045

    it = iteration if iteration == iteration and iteration >= 0 else 0
    of = overflow if overflow == overflow else 1.0
    of = 1.0 if of > 1.0 else (0.0 if of < 0.0 else of)

    # Exponential annealing baseline: high gamma early, low gamma late.
    base = GAMMA_LOW + (GAMMA_HIGH - GAMMA_LOW) * (E ** (-0.0015 * float(it)))

    # Overflow-adaptive: keep gamma smooth while bins are over-dense,
    # let it fall toward the accurate regime as overflow clears.
    base *= 0.45 + 0.55 * of

    # Stall detection: if overflow has stopped improving, sharpen gamma.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 6:
        recent = overflow_history[-6:]
        progress = float(recent[0]) - float(recent[-1])
        if progress < 1e-3:
            base *= 0.80

    # Damp against runaway gradients to avoid noisy, divergent steps.
    gn = gradient_norm if gradient_norm == gradient_norm and gradient_norm >= 0.0 else 0.0
    if gn > 1e4:
        base *= 0.90

    if base != base:
        base = 1.0
    return float(50.0 if base > 50.0 else (0.01 if base < 0.01 else base))