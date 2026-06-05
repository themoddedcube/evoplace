def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty multiplier with bounded growth.

    Grows lambda multiplicatively (DREAMPlace-style subgradient ascent on the
    density penalty), but the growth rate is modulated by how much overflow is
    still present and whether it is improving. Heavy overflow -> push harder;
    once cells have spread out -> ease off so wirelength can be fine-tuned.
    Everything is bounded to keep the multiplier well-conditioned and the
    returned value inside [0.01, 50.0] so the optimizer cannot diverge.
    """
    LOWER_PCOF = 1.03
    UPPER_PCOF = 1.06

    # Seed a sane value if the optimizer handed us something degenerate.
    if not (current_lambda > 0.0) or current_lambda != current_lambda:  # NaN/<=0 guard
        current_lambda = 0.01

    # Clamp overflow into [0, 1] defensively.
    of = overflow
    if of != of:                       # NaN
        of = 1.0
    of = 0.0 if of < 0.0 else (1.0 if of > 1.0 else of)

    # Trend: is overflow falling (good) or stuck/rising (need more pressure)?
    delta = 0.0
    if overflow_history:
        prev = overflow_history[-1]
        if prev == prev:               # not NaN
            delta = prev - of          # >0 means overflow improved

    # Base growth scales with remaining overflow: more overflow -> faster ramp.
    # of=1 -> UPPER_PCOF, of=0 -> LOWER_PCOF.
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # If overflow is stagnant or worsening, nudge growth up a touch; if it is
    # falling fast, relax toward 1.0 to stop over-penalizing density.
    if delta <= 0.0:
        mu += 0.01
    else:
        mu -= min(0.04, 4.0 * delta)
        if mu < 1.0:
            mu = 1.0

    # Late in the run (low overflow), stop growing lambda so HPWL can settle.
    if of < 0.10:
        mu = 1.0 - 0.5 * (0.10 - of)   # gentle decay toward fine-tuning
        if mu < 0.97:
            mu = 0.97

    # Guard against exploding gradients: if the gradient norm is very large,
    # do not amplify the penalty further this step.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e6 and mu > 1.0:
            mu = 1.0

    new_lambda = current_lambda * mu

    # Final hard clamp into the legal range.
    if new_lambda != new_lambda:       # NaN
        new_lambda = 0.01
    if new_lambda < 0.01:
        new_lambda = 0.01
    if new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)