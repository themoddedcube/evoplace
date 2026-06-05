def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    # Base geometric growth, decaying toward a floor as iterations progress.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push harder while bins are congested,
    # ease off as the layout legalizes so HPWL can refine.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow over recent history (negative => improving).
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-min(5, len(overflow_history)):]
        trend = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)

    if of > 0.10:
        # Still spreading: grow faster when overflow is high or stalling.
        stall_boost = 1.0 + (0.5 if trend >= -1e-4 else 0.0)
        mu = 1.0 + (base_mu - 1.0) * (0.6 + 1.4 * of) * stall_boost
    else:
        # Nearly legal: damp growth strongly to lock geometry and cut HPWL.
        mu = LOWER_PCOF * (0.5 + 5.0 * of)
        mu = max(mu, 0.985)   # allow slight relaxation for fine-tuning

    # Gradient-norm safeguard: avoid overshoot when gradients explode.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        mu = min(mu, 1.02)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))