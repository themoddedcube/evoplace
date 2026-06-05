def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative lambda update (ePlace-style).

    Grows the density penalty quickly while cells are still spread out
    (high overflow) and tapers the growth as the layout legalizes
    (overflow falling), so wirelength is not over-penalized late.
    """
    # Sanitize inputs so a NaN/inf can never propagate to inf output.
    if not (current_lambda == current_lambda) or current_lambda in (float("inf"), float("-inf")):
        current_lambda = 1.0
    of = overflow if (overflow == overflow) else 1.0
    of = min(max(of, 0.0), 1.0)

    # Base growth factor, decaying slowly with iteration (anneal the push).
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-trend term: accelerate while overflow is high or rising,
    # decelerate once it is falling and small.
    delta = 0.0
    if overflow_history:
        prev = overflow_history[-1]
        if prev == prev:  # not NaN
            delta = of - min(max(prev, 0.0), 1.0)

    # Map trend + level into a multiplier in [LOWER_PCOF, base].
    # Rising overflow (delta>0) or high level -> push harder.
    push = of + 5.0 * max(delta, 0.0)
    mu = LOWER_PCOF + (base - LOWER_PCOF) * min(push, 1.0)

    # If overflow has collapsed, stop growing (let WL gradients dominate).
    if of < 0.08:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Clamp to the required return range.
    if not (new_lambda == new_lambda):
        new_lambda = 1.0
    return float(min(max(new_lambda, 0.01), 50.0))