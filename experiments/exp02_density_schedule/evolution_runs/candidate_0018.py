def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    REF_OVERFLOW = 0.10

    # Base DREAMPlace-style growth: aggressive early, gentle late.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Recent overflow trend (negative => spreading is working).
    if len(overflow_history) >= 2:
        window = overflow_history[-min(5, len(overflow_history)):]
        delta = window[-1] - window[0]
    else:
        delta = 0.0

    if overflow > REF_OVERFLOW:
        # Cells still clustered: ramp the density weight, harder when stagnant.
        push = 1.0 + min(1.0, overflow - REF_OVERFLOW)
        if delta > -0.005:          # overflow not improving
            push *= 1.05
        mu = base * push
    else:
        # Near legalization: relax growth so wirelength can be fine-tuned.
        ease = max(LOWER_PCOF, 1.0 - (REF_OVERFLOW - overflow))
        mu = max(1.0, base * ease)

    # Scale-free gradient safeguard: damp the update only when the gradient is
    # large relative to lambda (prevents density-force blow-ups late in flow).
    if gradient_norm > 0.0 and current_lambda > 0.0:
        ratio = gradient_norm / (current_lambda + 1e-12)
        if ratio > 10.0:
            mu = 1.0 + (mu - 1.0) / (1.0 + 0.05 * (ratio - 10.0))

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))