def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    GAMMA_HIGH = 8.0
    GAMMA_LOW = 0.5

    # --- sanitize overflow (NaN/inf guard) and clamp to [0, 1] ---
    of = overflow
    if of != of or of in (float("inf"), float("-inf")):
        of = 1.0
    of = 1.0 if of > 1.0 else (0.0 if of < 0.0 else of)

    # smooth with recent overflow to suppress gradient noise
    if overflow_history:
        recent = [h for h in overflow_history[-5:] if h == h]
        if recent:
            of = 0.5 * of + 0.5 * (sum(recent) / len(recent))

    # --- overflow-adaptive core (DREAMPlace-style) ---
    # high overflow (cells spread/clustered early) -> high gamma -> smooth gradients
    # low overflow (near-legal late)               -> low gamma  -> accurate HPWL
    frac = (of - 0.1) / 0.9
    frac = 1.0 if frac > 1.0 else (0.0 if frac < 0.0 else frac)
    target = GAMMA_LOW * (10.0 ** (frac * 1.2))   # ~0.5 .. ~7.9

    # --- iteration safety floor: force annealing even if overflow stalls ---
    decay = 0.998 ** float(iteration)
    target = GAMMA_LOW + (target - GAMMA_LOW) * decay

    # --- damp transitions to avoid oscillation, but never trust inf state ---
    cl = current_lambda
    if cl == cl and 0.01 <= cl <= 50.0:
        new_lambda = 0.6 * cl + 0.4 * target
    else:
        new_lambda = target

    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)