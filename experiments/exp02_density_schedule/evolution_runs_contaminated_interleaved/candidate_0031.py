def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # --- sanitize overflow ------------------------------------------------
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # Target overflow we are driving toward (ePlace-style stop band).
    OF_TARGET = 0.07

    # --- base multiplier: ramp lambda up while density is bad -------------
    # RePlAce coupling: error above target pushes mu up, below target pulls
    # it down. Saturating map keeps it smooth and bounded.
    err = of - OF_TARGET
    drive = err / (abs(err) + 0.12)                 # in (-1, 1)
    coef = 1.0 + (UPPER_PCOF - 1.0) * max(drive, 0.0) \
               + (LOWER_PCOF - 1.0) * max(-drive, 0.0)

    # Gentle global decay so very late iterations stop over-pushing lambda.
    base = max(0.99985 ** float(iteration), 0.985)

    # --- overflow trajectory feedback (robust slope) ---------------------
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent                      # >0 means overflow falling
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))                     # in (-1, 1)
        if sat <= 0.0:
            # Overflow stalled/rising -> push harder to keep spreading down.
            coef *= 1.0 - 0.060 * sat
        else:
            # Healthy descent -> ease the push slightly, let HPWL settle.
            coef *= 1.0 - 0.040 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # --- near-target damping: don't overshoot the stop band --------------
    if of < 0.06:
        coef *= 0.86 + 1.0 * of
    elif of < 0.10:
        coef *= 0.95
    elif of < 0.18:
        coef *= 0.985

    # --- gradient guard: very large grads -> smaller step ----------------
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    mu = coef * base

    # --- adaptive clamp: wide early, tighten as we converge --------------
    prog = min(max((float(iteration) - 250.0) / 250.0, 0.0), 1.0)
    hi = 1.10 - 0.05 * prog
    lo = 0.90 + 0.04 * prog
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))