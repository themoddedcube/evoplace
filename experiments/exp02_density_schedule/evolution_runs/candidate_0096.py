def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Overflow-adaptive multiplicative growth of the density weight.
    # Grow fast while cells are still overlapping (high overflow),
    # ease off as the layout legalizes (low overflow), and damp the
    # step once overflow starts dropping so lambda doesn't overshoot.
    ov = overflow if overflow == overflow else 1.0          # guard NaN
    ov = min(max(ov, 0.0), 1.0)

    # Base per-iteration multiplier, larger when overflow is high.
    base = 1.0 + 0.06 * ov

    # Trend term: if overflow is falling, slow the growth; if it is
    # stuck or rising, push a little harder.
    trend = 0.0
    if overflow_history is not None and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if all(r == r for r in recent):
            trend = recent[-1] - recent[0]                  # >0 worsening
    base *= (1.0 + 0.5 * max(min(trend, 0.2), -0.2))

    # Gentle annealing of the growth rate so late iterations are stable.
    decay = max(0.97 ** float(iteration), 0.5)
    mu = 1.0 + (base - 1.0) * decay

    cur = current_lambda if (current_lambda == current_lambda) else 1.0
    if cur <= 0.0:
        cur = 0.01

    new_lambda = cur * mu

    # Keep finite and inside the required range.
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cur
    return float(min(max(new_lambda, 0.01), 50.0))