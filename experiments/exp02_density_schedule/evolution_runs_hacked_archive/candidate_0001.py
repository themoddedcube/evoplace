def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # DREAMPlace-style decaying base growth rate.
    decay = max(0.9999 ** float(iteration), 0.98)
    base_mu = UPPER_PCOF * decay

    # Overflow-adaptive scaling: drive the density weight hard while the
    # layout is congested, then taper growth toward 1.0 as overflow falls
    # so HPWL is not sacrificed during late fine-tuning.
    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = max(0.0, min(1.0, of))
    ramp = min(1.0, of / 0.15)                              # full growth above 15% overflow
    mu = 1.0 + (base_mu - 1.0) * ramp

    # Stagnation detection: if overflow has stalled while still high, give
    # the penalty an extra nudge to break out of the plateau.
    if len(overflow_history) >= 3 and of > 0.10:
        recent = overflow_history[-3:]
        if (recent[0] - recent[-1]) < 1e-3:
            mu *= 1.03

    # Ease off once nearly legalized to protect wirelength.
    if of < 0.05:
        mu = 1.0 + (mu - 1.0) * (of / 0.05)

    # Clamp the per-step multiplier to a safe band.
    mu = max(LOWER_PCOF, min(UPPER_PCOF * 1.05, mu))

    new_lambda = current_lambda * mu
    return float(max(0.01, min(50.0, new_lambda)))