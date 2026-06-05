def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty (lambda) schedule.

    Grows lambda multiplicatively (DREAMPlace style) but modulates the
    growth rate by overflow and overflow trend, and hard-clamps the
    result to the legal range so the schedule never diverges.
    """
    LO, HI = 0.01, 50.0

    # Defensive handling of degenerate inputs.
    lam = current_lambda
    if not (lam == lam) or lam <= 0.0:   # NaN or non-positive
        lam = 1.0
    lam = min(max(lam, LO), HI)

    ovf = overflow
    if not (ovf == ovf):                 # NaN guard
        ovf = 1.0
    ovf = min(max(ovf, 0.0), 1.0)

    # Base multiplicative growth, annealed slowly with iteration so the
    # penalty ramps hard early (spread cells) and gently late (fine-tune).
    anneal = max(0.9999 ** float(iteration), 0.95)
    base_mu = 1.0 + 0.05 * anneal

    # Overflow-adaptive boost: push harder while many bins are overfull,
    # ease off as the layout legalizes so HPWL can settle.
    #   ovf ~ 1.0 -> ~ +1.5x extra growth pressure
    #   ovf ~ 0.0 -> mild decay toward fine-tuning
    overflow_gain = 1.0 + 1.0 * (ovf - 0.10)

    # Trend term: if overflow has stalled (not decreasing), increase the
    # penalty more aggressively to break out of the plateau.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[0] - recent[-1]   # positive => overflow improving
        if delta < 1e-4:                 # stalled or worsening
            trend = 0.04
        elif delta > 0.02:               # improving fast, relax growth
            trend = -0.02

    mu = base_mu * overflow_gain + trend

    # Keep the per-step multiplier sane to avoid runaway / collapse.
    mu = min(max(mu, 0.97), 1.20)

    new_lambda = lam * mu

    # When the placement is essentially legal, gently relax the penalty
    # to let the wirelength term dominate the final iterations.
    if ovf < 0.05:
        new_lambda *= 0.99

    return float(min(max(new_lambda, LO), HI))