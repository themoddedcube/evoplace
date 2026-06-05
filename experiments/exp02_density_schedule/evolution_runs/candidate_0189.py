def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplicative growth, decaying with iteration (DREAMPlace-style)
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive coefficient: push harder while spreading is poor,
    # ease off as the layout legalizes so we don't over-penalize density.
    of = overflow if overflow == overflow else 1.0  # NaN guard
    of = min(max(of, 0.0), 1.0)
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Stagnation detection: if overflow stops improving, accelerate growth
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-3 and of > 0.1:
            pcof *= 1.03  # nudge density weight up to break the plateau
        elif improvement < 0.0:
            pcof *= 1.02  # overflow worsening -> push density harder

    # Gradient-norm safeguard: if gradients explode, temper the increase
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            pcof = min(pcof, 1.01)

    mu = pcof * decay

    new_lambda = current_lambda * mu
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)