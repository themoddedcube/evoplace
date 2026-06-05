def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LOWER = 0.01
    UPPER = 50.0

    def _finite(x, default):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return default
        if xf != xf or xf == float("inf") or xf == float("-inf"):
            return default
        return xf

    # Sanitize inputs so a bad upstream value never yields inf/nan output.
    of = min(max(_finite(overflow, 1.0), 0.0), 1.0)
    lam = min(max(_finite(current_lambda, 1.0), LOWER), UPPER)

    # Base geometric growth of the density weight, gently annealed over iters.
    UPPER_PCOF = 1.05
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: high overflow -> push spreading harder;
    # low overflow (cells settled) -> grow slowly so HPWL can fine-tune.
    boost = 1.0 + 0.10 * of

    # Stagnation: if overflow stopped improving, raise density pressure.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        recent = [_finite(v, of) for v in overflow_history[-3:]]
        if (recent[0] - recent[-1]) < 1e-3:
            boost *= 1.05

    # Keep the per-step multiplier in a stable band to avoid divergence.
    mu = min(max(base * boost, 0.5), 1.5)

    new_lambda = lam * mu
    if new_lambda != new_lambda:  # nan guard
        new_lambda = lam

    return float(min(max(new_lambda, LOWER), UPPER))