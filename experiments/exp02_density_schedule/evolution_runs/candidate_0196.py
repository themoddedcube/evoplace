def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Multiplicative density-weight update (DREAMPlace-style), but the growth
    # rate adapts to overflow progress instead of a fixed geometric decay.
    # High overflow (cells still spread) -> push lambda up faster so density
    # forces clustering; low overflow (nearly legal) -> ease off so HPWL can
    # be fine-tuned without density over-penalizing.

    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)

    # Base per-iteration growth, annealed so late iterations move gently.
    anneal = max(0.9995 ** float(iteration), 0.95)

    # Overflow-adaptive multiplier: large step while far from target overflow,
    # shrinking toward ~1.0 (and slightly below) as the layout legalizes.
    TARGET_OF = 0.10
    drive = (of - TARGET_OF) / max(1.0 - TARGET_OF, 1e-6)    # ~1 early, <0 once legal
    drive = min(max(drive, -1.0), 1.0)

    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    mu = 1.0 + drive * (UPPER_PCOF - 1.0) if drive >= 0.0 \
        else 1.0 + drive * (1.0 - LOWER_PCOF)
    mu *= anneal

    # Trend damping: if overflow is rising (placement diverging), pull mu back.
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev and of > prev + 1e-4:
            mu *= 0.98

    # Gradient-norm safety: if gradients blow up, don't amplify the penalty.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))