def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive gamma: smooth (high) while clustered, sharp (low) when spread."""
    GAMMA_HI = 8.0   # smooth gradients early (cells clustered, overflow ~1)
    GAMMA_LO = 0.5   # accurate HPWL late (cells spread, overflow ~0)

    # Sanitize overflow into [0, 1]; treat NaN/garbage as "early, fully clustered".
    ovf = overflow
    if ovf != ovf or ovf < 0.0:
        ovf = 1.0
    elif ovf > 1.0:
        ovf = 1.0

    # Exponential interpolation in log-space: ovf=1 -> GAMMA_HI, ovf=0 -> GAMMA_LO.
    # This is the canonical DREAMPlace-style overflow-driven gamma annealing.
    gamma = GAMMA_LO * (GAMMA_HI / GAMMA_LO) ** ovf

    # Stagnation detection: if overflow has plateaued, the coarse phase is done —
    # sharpen the WA-WL approximation to push true HPWL down.
    if overflow_history is not None and len(overflow_history) >= 5:
        recent = overflow_history[-5:]
        if (recent[0] - recent[-1]) < 1e-3:
            gamma *= 0.8

    # Gradient safety: if gradients blow up, raise gamma to smooth and stabilize.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        gamma *= 1.5

    # Guaranteed fine-tuning floor in the late phase, independent of overflow noise.
    if iteration > 500:
        if gamma > 1.0:
            gamma = 1.0

    # Final clamp to the allowed range with NaN guard.
    if gamma != gamma:
        gamma = GAMMA_LO
    if gamma < 0.01:
        gamma = 0.01
    elif gamma > 50.0:
        gamma = 50.0
    return float(gamma)