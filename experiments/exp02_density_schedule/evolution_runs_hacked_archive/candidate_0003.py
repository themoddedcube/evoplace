def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Iteration-decayed base growth (geometric, as in ePlace/DREAMPlace).
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow trend: improving (decreasing) is good and analogous to delta_HPWL < 0.
    if len(overflow_history) >= 2:
        delta = float(overflow_history[-1]) - float(overflow_history[-2])
    else:
        delta = -1.0

    of = min(max(float(overflow), 0.0), 1.0)

    if delta < 0.0:
        # Overflow still falling: hold near the nominal growth rate, but ease off
        # once density is nearly resolved so wirelength can be fine-tuned.
        decay = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of
        mu = decay * base
    else:
        # Overflow stalled or rising: accelerate the density penalty to break the
        # plateau, scaled by how badly it regressed.
        ratio = min(delta / 0.01, 1.0)
        mu = UPPER_PCOF * (1.0 + 0.5 * ratio) * base

    # Damp updates when gradients are very noisy (small gradient_norm) to avoid
    # over-driving lambda in the late, low-gamma regime.
    if gradient_norm > 0.0 and gradient_norm < 1e-3:
        mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))