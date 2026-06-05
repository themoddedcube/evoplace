def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # DREAMPlace-style adaptive density-weight (lambda) ramp.
    # Core idea: grow lambda geometrically, but modulate the growth rate by
    # how the overflow is actually evolving rather than by iteration alone.
    #  - overflow still high / stagnating  -> push lambda up faster (spread cells)
    #  - overflow dropping nicely          -> ease off so wirelength can settle
    # All multipliers are kept mild and the result is hard-clamped to [0.01, 50.0].

    LOWER_PCOF = 1.03
    UPPER_PCOF = 1.06

    # Base geometric schedule (annealed toward the lower bound late in the run),
    # mirroring the original mu but with a gentler late-iteration floor.
    base = max(0.9999 ** float(iteration), 0.985)

    # Estimate the recent overflow trend from history (negative = improving).
    trend = 0.0
    if overflow_history is not None and len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        trend = recent[-1] - recent[0]
    elif overflow_history is not None and len(overflow_history) >= 2:
        trend = overflow_history[-1] - overflow_history[0]

    # Map trend into [0, 1]: ~1 when overflow is stuck/rising, ~0 when falling fast.
    # 0.02 sets the sensitivity to per-window overflow change.
    if trend >= 0.0:
        stagnation = 1.0
    else:
        stagnation = max(0.0, 1.0 + trend / 0.02)

    # Overflow level itself: more aggressive while the layout is still dense.
    level = overflow if overflow == overflow else 1.0  # guard against NaN
    level = min(max(level, 0.0), 1.0)

    # Blend stagnation and absolute level to pick a per-iteration multiplier.
    drive = 0.5 * stagnation + 0.5 * level
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * drive

    # Dampen the push when gradients are exploding to avoid destabilizing the solve.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        damp = 1.0 / (1.0 + 0.05 * max(0.0, gradient_norm - 1.0))
        pcof = 1.0 + (pcof - 1.0) * damp

    mu = pcof * base

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:  # NaN safety
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))