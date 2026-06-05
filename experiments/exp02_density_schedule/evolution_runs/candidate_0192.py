def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    TARGET_OVERFLOW = 0.10

    # DREAMPlace-style base multiplier: aggressive early, decaying toward ~1 late.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: spread cells hard while bins are over-dense,
    # then ease off as overflow approaches the target so HPWL can settle.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)
    if of > TARGET_OVERFLOW:
        of_factor = 1.0 + 0.6 * min(of - TARGET_OVERFLOW, 0.5)
    else:
        of_factor = max(0.5, of / TARGET_OVERFLOW)

    mu = base * of_factor

    # Trend awareness: detect stagnation vs. oscillation over recent history.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 5:
        recent = [float(x) for x in overflow_history[-5:]]
        improving = recent[0] - recent[-1]
        if improving <= 1e-4 and of > TARGET_OVERFLOW:
            # Overflow plateaued while still too dense -> push harder.
            mu *= 1.08
        elif improving > 0.05:
            # Rapid spreading -> back off to avoid overshoot/divergence.
            mu *= 0.97
        # Oscillation damping.
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        sign_flips = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
        if sign_flips >= 2:
            mu = 1.0 + 0.5 * (mu - 1.0)

    # Gradient-norm safeguard: if gradients explode, dampen the update.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.0 + 0.5 * (mu - 1.0))

    # Clamp the per-step multiplier to prevent runaway growth (the inf failure mode).
    mu = min(max(mu, LOWER_PCOF), 1.25)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))