def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # DREAMPlace-style multiplicative density-weight schedule with
    # overflow-adaptive growth rate. lambda grows fast while cells are
    # still spread out (high overflow) and tapers as the layout legalizes.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    of = overflow if overflow == overflow else 1.0   # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Base decay of the growth multiplier so late iterations are gentle.
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push harder when overflow is high, ease off
    # as the placement becomes legal so we stop over-penalizing density.
    of_factor = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Watch the overflow trend: if overflow has stalled (not decreasing),
    # increase pressure; if it is dropping nicely, relax the multiplier.
    if overflow_history and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[0] - recent[-1]          # positive => improving
        if delta < 1e-4:                        # stalled
            of_factor *= 1.02
        elif delta > 0.02:                      # improving fast
            of_factor *= 0.99

    # Damp growth if gradients are exploding to keep optimization stable.
    gn = gradient_norm if gradient_norm == gradient_norm else 1.0
    if gn > 1e3:
        of_factor = 1.0 + (of_factor - 1.0) * 0.5

    mu = min(of_factor, base * UPPER_PCOF)
    mu = max(mu, 1.0)                            # never shrink lambda

    cl = current_lambda if current_lambda == current_lambda else 1.0
    new_lambda = cl * mu

    return float(min(max(new_lambda, 0.01), 50.0))