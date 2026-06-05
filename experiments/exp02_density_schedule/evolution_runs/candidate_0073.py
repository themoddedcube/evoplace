def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive, bounded density-penalty growth."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    # NaN/inf guards on inputs
    cur = current_lambda
    if cur != cur or cur in (float("inf"), float("-inf")):
        cur = 1.0
    of = overflow
    if of != of:
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    # DREAMPlace-style decaying base step, never below 0.98
    base = max(0.9999 ** float(iteration), 0.98)

    # Push the penalty harder while overflow is high (cells still overlapping),
    # ease the step toward 1.0 as the layout legalizes.
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of
    mu = pcof * base

    # Stagnation: if recent overflow stopped improving, accelerate slightly.
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        if recent[-1] >= recent[0] - 1e-4 and of > 0.1:
            mu *= 1.02

    # Near convergence, stop inflating lambda so HPWL can be fine-tuned.
    if of < 0.08:
        mu = min(mu, 1.0)

    # Gradient explosion guard: damp growth when gradients blow up.
    gn = gradient_norm
    if gn == gn and gn > 1e3:
        mu = min(mu, 1.0)

    new_lambda = cur * mu
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cur

    return float(min(max(new_lambda, 0.01), 50.0))