def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule (DREAMPlace-style),
    with hard clamping to keep the multiplier well-conditioned."""
    LOWER, UPPER = 0.01, 50.0

    # Base geometric growth, annealed so early iterations push harder
    # on the density penalty and later iterations settle.
    base = 1.05 * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive correction: when bins are still very congested we
    # want a stronger push on the density weight; once overflow drops we
    # ease off so wirelength can be fine-tuned without overshoot.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend from recent history: accelerate if overflow is stalling,
    # decelerate if it is already falling quickly.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-1]
        if prev == prev:                                    # guard NaN
            trend = of - min(max(prev, 0.0), 1.0)

    # Map overflow level to a multiplicative factor in roughly [0.97, 1.06]:
    # high overflow -> >1 (grow), low overflow -> ~1 (hold/relax).
    adapt = 1.0 + 0.06 * of - 0.03 * (1.0 - of)
    if trend > 0.0:                                         # overflow stalled/rising
        adapt *= 1.0 + min(trend, 0.05)
    elif trend < 0.0:                                       # overflow improving
        adapt *= 1.0 - min(-trend, 0.03)

    mu = base * adapt

    # Keep the per-step multiplier sane so lambda neither explodes nor collapses.
    mu = min(max(mu, 0.95), 1.10)

    nxt = current_lambda * mu

    # Hard clamp to the legal range; also guard against NaN/inf inputs.
    if not (nxt == nxt) or nxt in (float("inf"), float("-inf")):
        nxt = current_lambda if (current_lambda == current_lambda) else 1.0
    return float(min(max(nxt, LOWER), UPPER))