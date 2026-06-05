def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LOWER, UPPER = 0.01, 50.0

    # --- sanitize current lambda (guard against NaN/inf/non-positive) ---
    lam = current_lambda
    if lam != lam or lam in (float("inf"), float("-inf")) or lam <= 0.0:
        lam = 1.0
    lam = min(max(lam, LOWER), UPPER)

    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # --- overflow trend (are cells still spreading out?) ---
    if overflow_history and len(overflow_history) >= 1:
        prev = overflow_history[-1]
        delta = of - prev  # negative = legalizing
    else:
        delta = 0.0

    # --- DREAMPlace-style adaptive penalty multiplier ---
    # Grow the density weight harder while overflow is high; ease off as
    # the layout legalizes so we don't overshoot and corrupt wirelength.
    LOWER_PCOF, UPPER_PCOF = 0.95, 1.05
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # If overflow stalls while still high, push harder to break the plateau.
    if delta > -1e-4 and of > 0.10:
        mu *= 1.0 + min(of, 0.5)

    # Near convergence: hold lambda steady to let wirelength fine-tune.
    if of < 0.05:
        mu = 1.0

    new_lam = lam * mu
    if new_lam != new_lam:  # NaN guard
        new_lam = lam
    return float(min(max(new_lam, LOWER), UPPER))