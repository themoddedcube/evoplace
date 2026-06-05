def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Baseline DREAMPlace-style multiplicative growth that decays with
    # iteration so lambda ramps quickly early and gently later.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    of = overflow
    if of < 0.0:
        of = 0.0
    elif of > 1.0:
        of = 1.0

    # Overflow-adaptive scaling. While density is far from legal, push
    # lambda harder to spread cells; as overflow nears the target, relax
    # growth so the (low-gamma) wirelength gradients can fine-tune HPWL.
    ref = 0.10
    if of > ref:
        accel = 1.0 + 0.6 * (of - ref)          # proportional push
        mu = base * accel
    else:
        # Smoothly fade the increment to ~1.0 in the endgame.
        frac = of / ref
        mu = 1.0 + (base - 1.0) * frac

    # Stagnation escape: if overflow has stopped improving while still
    # above target, give lambda an extra nudge to break the plateau.
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-3 and of > ref:
            mu *= 1.03
        # Overshoot guard: overflow already collapsed and rising lambda
        # would only hurt wirelength — back off toward neutral.
        elif of <= ref and improvement <= 0.0:
            mu = min(mu, 1.0 + 0.25 * (base - 1.0))

    # Numerical guards on the multiplier itself.
    if mu < LOWER_PCOF:
        mu = LOWER_PCOF
    elif mu > UPPER_PCOF * 1.5:
        mu = UPPER_PCOF * 1.5

    new_lambda = current_lambda * mu

    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return new_lambda