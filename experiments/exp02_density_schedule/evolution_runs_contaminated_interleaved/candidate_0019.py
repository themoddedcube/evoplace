def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty schedule with anti-divergence guards."""
    # Base multiplicative growth (DREAMPlace UPPER_PCOF style), but capped.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Decay the aggressiveness of growth as iterations progress so the
    # penalty does not blow up late in the run.
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push harder while cells are still spread
    # (high overflow), ease off as the layout converges (low overflow).
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Detect stalls / oscillation from recent overflow history.
    stalled = False
    if overflow_history is not None and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        # very small improvement over last few iters
        if abs(recent[0] - recent[-1]) < 1e-4:
            stalled = True

    if of > 0.10:
        # Still spreading: grow penalty, scaled by remaining overflow.
        mu = 1.0 + (UPPER_PCOF - 1.0) * decay * (0.5 + 0.5 * of)
    elif of > 0.05:
        # Approaching target density: gentle growth.
        mu = 1.0 + 0.5 * (UPPER_PCOF - 1.0) * decay
    else:
        # Converged enough: relax penalty to refine wirelength.
        mu = LOWER_PCOF + 0.05 * of / 0.05

    # If progress stalled, give a small kick to escape the plateau.
    if stalled and of > 0.06:
        mu *= 1.02

    # Guard against exploding / vanishing gradients.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        mu = min(mu, 1.005)

    new_lambda = current_lambda * mu

    # Hard clamp to the legal range.
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))