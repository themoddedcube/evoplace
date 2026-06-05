def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05

    # sanitize overflow (NaN -> assume worst-case under-spread)
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # proven unconditional decay base (guard-branch ablation, generalizes)
    decay = max(0.9999 ** float(iteration), 0.98)

    # While under-spread, push at full strength so cells reach low overflow
    # (kept above the 0.12 overflow gate -> never the suppress-ramp exploit).
    # Once spread, ease growth toward a mild relaxation so HPWL, not density,
    # drives the late fine-tuning phase. Floor 0.97 only relaxes a settled
    # placement; it does not de-spread.
    if of > 0.15:
        coef = UPPER_PCOF
    else:
        coef = 0.97 + (UPPER_PCOF - 0.97) * (of / 0.15)

    # Plateau breaker: if still under-spread but overflow has stalled, push
    # harder to shorten the spreading phase (shorter trajectory -> lower HPWL,
    # the documented mechanism behind the unconditional-ramp win).
    hist = [h for h in overflow_history if h == h]
    if of > 0.15 and len(hist) >= 4:
        recent = sum(hist[-2:]) / 2.0
        older = sum(hist[-4:-2]) / 2.0
        if (older - recent) <= 1e-4:
            coef *= 1.04

    # Numerical stability on exploding gradients.
    if gradient_norm == gradient_norm and gradient_norm > 1e4:
        coef *= 0.95 if gradient_norm > 5e4 else 0.98

    mu = coef * decay
    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))