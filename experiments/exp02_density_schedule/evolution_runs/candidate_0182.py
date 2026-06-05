def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Sanitize inputs (guard against NaN/inf that produce inf HPWL).
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    cur = _finite(current_lambda, 1.0)
    ovfl = _finite(overflow, 1.0)
    gnorm = _finite(gradient_norm, 1.0)
    cur = min(max(cur, 0.01), 50.0)
    ovfl = min(max(ovfl, 0.0), 1.0)

    # Base subgradient-style growth (DREAMPlace UPPER/LOWER bounded).
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: push density weight harder while spread out,
    # ease off as the layout legalizes so HPWL can be fine-tuned.
    if overflow_history and len(overflow_history) >= 2:
        prev = _finite(overflow_history[-2], ovfl)
        delta = ovfl - prev
    else:
        delta = 0.0

    if ovfl > 0.10:
        # Still overflowing: accelerate, but more when overflow is stalling.
        stall = 1.0 if delta >= -1e-4 else 0.5
        mu = base_mu * (1.0 + 0.6 * ovfl * stall)
    else:
        # Nearly legal: decay the weight toward gentle fine-tuning.
        mu = LOWER_PCOF + 0.10 * (ovfl / 0.10)

    # Gradient-norm safety: if gradients explode, damp growth to avoid divergence.
    if gnorm > 1e3:
        mu = min(mu, 1.0)

    new_lambda = cur * mu
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cur
    return min(max(new_lambda, 0.01), 50.0)