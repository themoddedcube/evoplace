def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative lambda schedule for DREAMPlace.

    Grows the density penalty geometrically (augmented-Lagrangian style) but
    modulates the growth rate by how legalized the layout already is:
      - high overflow  -> push lambda up harder to spread cells
      - low overflow   -> ease off so HPWL can settle without over-penalizing
    A trend term reacts to whether overflow is improving, and the per-step
    multiplier is annealed toward 1 so late iterations fine-tune gently.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Base per-step multiplier, annealed toward 1.0 as iterations progress
    # (matches the original's decaying-mu behavior, but floored higher).
    base_mu = max(0.9999 ** float(iteration), 0.98)

    # Overflow trend: positive when overflow is dropping (good).
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-1]
        past = overflow_history[-min(5, len(overflow_history))]
        if recent == recent and past == past:
            trend = past - recent  # >0 means improving

    # Map overflow to a target multiplier in [LOWER_PCOF, UPPER_PCOF].
    # When far from legal (of high) -> near UPPER_PCOF; when nearly legal -> ~1.0.
    span = UPPER_PCOF - 1.0
    pcof = 1.0 + span * of

    # If overflow is improving fast, relax the push a touch; if it's stuck or
    # worsening while still congested, push a bit harder.
    if trend > 0.005:
        pcof -= 0.02 * min(trend / 0.02, 1.0)
    elif of > 0.1:
        pcof += 0.02 * min((-trend) / 0.02 if trend < 0 else 0.0, 1.0)

    # Gentle gradient-norm damping: huge gradients -> back off to stay stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            pcof = 1.0 + (pcof - 1.0) * 0.5

    pcof = min(max(pcof, LOWER_PCOF), UPPER_PCOF)

    mu = pcof * base_mu

    new_lambda = current_lambda * mu

    return float(min(max(new_lambda, 0.01), 50.0))