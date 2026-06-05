def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """ ... """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base multiplicative growth (DREAMPlace-style), annealed with iteration.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: when overflow is high, cells are still
    # spread out, so push the density penalty harder; as overflow drops the
    # placement is nearly legal, so grow gently to avoid disturbing HPWL.
    of = overflow if overflow == overflow else 1.0           # guard NaN
    of = min(max(of, 0.0), 1.0)
    adapt = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.5)

    # Detect stagnation: if overflow has stopped improving, accelerate the
    # penalty to break out of the plateau.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-3:
            adapt *= 1.02

    mu = 0.5 * base + 0.5 * adapt

    # Damp the multiplier when gradients explode to keep optimization stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = current_lambda * mu
    return min(max(new_lambda, 0.01), 50.0)