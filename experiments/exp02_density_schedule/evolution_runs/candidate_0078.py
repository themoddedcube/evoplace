def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # DREAMPlace-style multiplicative density-weight update, but made
    # overflow-adaptive and numerically robust so lambda never blows up
    # (which is what drives HPWL -> inf) nor stalls before legalization.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.003

    of = overflow if overflow == overflow else 1.0          # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Base geometric decay of the growth rate (same spirit as the original mu),
    # so early iterations push density hard and later ones ease off.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: when overflow is still high we want a
    # stronger density push; as overflow collapses we shrink the growth toward
    # ~1.0 so the placement can fine-tune wirelength without over-penalizing.
    adapt = LOWER_PCOF + (base - LOWER_PCOF) * of

    # Trend term: if overflow is plateauing (not decreasing), nudge growth up a
    # touch to break the stall; if it is dropping fast, hold back to stay stable.
    if overflow_history is not None and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        drop = recent[0] - recent[-1]
        if drop < 1e-4:
            adapt *= 1.01          # plateau -> push a little harder
        elif drop > 0.05:
            adapt *= 0.99          # rapid descent -> damp to avoid overshoot

    # Gradient safety: if gradients are exploding, damp the growth so the
    # optimizer stays in a stable regime.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        adapt = min(adapt, 1.01)

    mu = min(max(adapt, 0.95), UPPER_PCOF)

    next_lambda = current_lambda * mu
    return float(min(max(next_lambda, 0.01), 50.0))