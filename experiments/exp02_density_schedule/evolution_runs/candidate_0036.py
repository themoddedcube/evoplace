def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight multiplier (ePlace/RePlAce style)
    with a late-run annealing floor and hard stability clamps."""
    # --- sanitize inputs (NaN/Inf guards) ---
    of = overflow if overflow == overflow else 1.0
    if of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    cur = current_lambda
    if cur != cur or cur in (float("inf"), float("-inf")):
        cur = 0.01
    cur = min(max(cur, 0.01), 50.0)

    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # --- overflow trend from history ---
    delta = 0.0
    if overflow_history:
        prev = overflow_history[-1]
        if prev == prev and prev not in (float("inf"), float("-inf")):
            delta = of - min(max(prev, 0.0), 1.0)

    # Grow lambda while overflow is high; grow faster when overflow is falling
    # (negative delta), back off when it rises again.
    mu = 1.0 + 0.10 * of - 1.5 * delta

    # Gradient-norm safety: if gradients blow up, stop pushing density harder.
    gn = gradient_norm
    if gn == gn and gn not in (float("inf"), float("-inf")) and gn > 1e6:
        mu = min(mu, 1.0)

    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    # --- anneal aggressiveness late so converged placement isn't perturbed ---
    anneal = max(0.9999 ** float(iteration), 0.98)
    mu = 1.0 + (mu - 1.0) * anneal

    new_lambda = cur * mu

    # --- hard clamp to the allowed range ---
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cur
    return float(min(max(new_lambda, 0.01), 50.0))