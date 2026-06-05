def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Overflow improvement over the last step (positive = density getting better)
    if len(overflow_history) >= 2:
        delta = overflow_history[-2] - overflow_history[-1]
    else:
        delta = 0.0

    of = min(max(overflow, 0.0), 1.0)

    # Reference improvement scales with remaining overflow: expect bigger drops
    # while far from a legal placement, smaller drops near convergence.
    ref = 0.02 * max(of, 0.05)
    ratio = delta / ref

    # RePlAce-style adaptive multiplier: grow lambda gently while density is
    # improving on schedule, accelerate the penalty when progress stalls.
    mu = UPPER_PCOF ** (1.0 - ratio)
    mu = min(max(mu, LOWER_PCOF), 1.10)

    # Warm-up: keep gradients smooth early (cells still clustering) by easing
    # the penalty growth in over the first iterations.
    warm = min(1.0, float(iteration) / 30.0)
    mu = 1.0 + (mu - 1.0) * warm

    # As overflow collapses, stop inflating lambda so wirelength can relax
    # during fine-tuning instead of being over-penalized by density.
    if of < 0.1:
        mu = min(mu, 1.0 + 0.2 * of)

    # Stability guard: damp growth if gradients are exploding.
    if gradient_norm > 0.0 and gradient_norm > 1e4:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))