def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Density-weight schedule for differentiable global placement.
    # Strategy: grow the density penalty multiplicatively, but make the
    # growth rate overflow-adaptive and clamp the result so it never
    # blows up (which is what produces inf HPWL on the unclamped version).

    # Guard against degenerate inputs.
    if current_lambda != current_lambda or current_lambda <= 0.0:  # NaN or non-positive
        current_lambda = 0.01
    ov = overflow if (overflow == overflow) else 1.0
    ov = min(max(ov, 0.0), 1.0)

    # Base multiplicative step (DREAMPlace-style), decaying with iteration so
    # early iterations ramp the penalty quickly and late iterations settle.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    decay = max(0.9999 ** float(iteration), 0.985)

    # Overflow-adaptive blend: when overflow is high the layout is still
    # spread out, so push the penalty harder; as overflow drops toward the
    # target, ease off to let wirelength fine-tune without overshoot.
    TARGET_OVERFLOW = 0.07
    if ov > TARGET_OVERFLOW:
        # how aggressively to climb, scaled by remaining overflow
        progress = min((ov - TARGET_OVERFLOW) / (1.0 - TARGET_OVERFLOW), 1.0)
        pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * progress
    else:
        # near/under target density: relax penalty slightly to recover HPWL
        pcof = 0.98

    # Detect stagnation in overflow history and nudge harder to break plateaus.
    if isinstance(overflow_history, list) and len(overflow_history) >= 5:
        recent = [h for h in overflow_history[-5:] if h == h]
        if len(recent) >= 5:
            spread = max(recent) - min(recent)
            if spread < 1e-3 and ov > TARGET_OVERFLOW:
                pcof *= 1.02  # plateau: extra push

    mu = pcof * decay

    new_lambda = current_lambda * mu

    # Hard clamp to the legal range; this is the key fix vs. the unbounded
    # baseline that diverged to inf.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0

    return float(new_lambda)