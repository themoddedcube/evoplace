def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Overflow-adaptive multiplicative penalty schedule.
    # Grow lambda fast while bins are congested, slow as the layout legalizes,
    # and damp growth when gradients blow up (prevents the divergence -> inf).

    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Base step: classic DREAMPlace UPPER_PCOF with iteration decay.
    UPPER_PCOF, LOWER_PCOF = 1.05, 1.001
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow shaping: high overflow -> push harder (toward UPPER),
    # low overflow -> ease off (toward LOWER) so HPWL can be fine-tuned.
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.5)
    mu *= decay

    # Trend awareness: if overflow is stalling/rising, nudge growth up;
    # if dropping nicely, let it relax.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev:
            if of > prev - 1e-4:        # not improving
                mu *= 1.02
            else:                       # improving
                mu *= 0.99

    # Gradient safety: damp when gradients are large/non-finite to avoid blow-up.
    gn = gradient_norm if gradient_norm == gradient_norm else 1.0
    if gn > 1e4 or gn != gn:
        mu = min(mu, 1.0)
    mu = min(max(mu, 0.5), 1.10)

    cl = current_lambda if (current_lambda == current_lambda and current_lambda > 0.0) else 1.0
    new_lambda = cl * mu

    if new_lambda != new_lambda:        # final NaN guard
        new_lambda = cl
    return float(min(max(new_lambda, 0.01), 50.0))