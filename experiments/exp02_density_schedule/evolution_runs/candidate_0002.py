def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Density-weight (lambda) update for differentiable global placement.
    # Grow lambda multiplicatively to spread cells, but modulate the growth
    # rate by overflow progress so the penalty never overshoots and diverges
    # (unbounded multiplicative growth was what sent HPWL to inf).
    base = current_lambda if current_lambda > 1e-6 else 0.01

    # Overflow trend: positive delta => overflow dropping (we are spreading).
    delta = 0.0
    if overflow_history:
        delta = float(overflow_history[-1]) - overflow

    # Base multiplier: push hardest while overflow is high, relax toward 1.0
    # as overflow falls so lambda settles and HPWL can be fine-tuned.
    drive = overflow if overflow < 1.0 else 1.0
    if drive < 0.0:
        drive = 0.0
    mu = 1.0 + 0.05 * drive

    # Stalled but still congested -> nudge harder to escape the plateau.
    if overflow > 0.10 and delta <= 1e-4:
        mu += 0.03

    # Overflow rising (diverging) -> back the penalty growth off.
    if delta < -1e-4:
        mu = max(1.0, mu - 0.04)

    # Exploding gradients -> damp growth to stay numerically stable.
    if gradient_norm > 1e3:
        mu = min(mu, 1.01)

    new_lambda = base * mu

    # Hard clamp to the legal range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)