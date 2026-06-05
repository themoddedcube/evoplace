def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-weight schedule with damping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Sanitize inputs.
    it = max(0, int(iteration))
    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    cl = current_lambda if current_lambda == current_lambda else 1.0
    cl = min(max(cl, 0.01), 50.0)

    # Base multiplier: strong early push that anneals toward 1.0 so the
    # density weight stops growing once cells are well spread (high gamma
    # early -> fine-tuning late).
    anneal = max(0.9999 ** float(it), 0.98)
    base = UPPER_PCOF * anneal

    # Overflow adaptation: push harder while many bins are over-dense,
    # ease off as overflow falls so HPWL can be refined accurately.
    # of ~ 1.0 -> ~base; of ~ 0.0 -> mild growth toward 1.0.
    overflow_gain = LOWER_PCOF + (base - LOWER_PCOF) * of

    # Trend term: if overflow is no longer improving, accelerate slightly;
    # if it is dropping fast, relax to avoid overshoot/oscillation.
    trend = 1.0
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        prev = overflow_history[-3]
        if prev == prev:
            delta = of - prev                              # >0 => stalling/worse
            trend = 1.0 + max(-0.03, min(0.05, delta))

    mu = overflow_gain * trend

    # Gradient safety: damp updates when gradients blow up (prevents the
    # divergence that drove HPWL to inf).
    if gn > 1e3:
        mu = 1.0 + (mu - 1.0) * (1e3 / gn)

    # Keep the per-step multiplier sane.
    mu = min(max(mu, 0.90), 1.10)

    new_lambda = cl * mu
    return float(min(max(new_lambda, 0.01), 50.0))