def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Overflow-adaptive multiplicative density-weight update (DREAMPlace-style),
    # but with bounded, stall-aware step size and hard clamping to avoid divergence.
    LO, HI = 0.01, 50.0

    # Sanitize inputs (NaN/inf -> safe defaults) without imports.
    def _finite(x, default):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return default
        # NaN != itself; inf comparisons below catch the rest.
        if xf != xf or xf == float("inf") or xf == float("-inf"):
            return default
        return xf

    of = _finite(overflow, 1.0)
    of = min(max(of, 0.0), 1.0)
    cur = _finite(current_lambda, LO)
    cur = min(max(cur, LO), HI)
    gnorm = _finite(gradient_norm, 1.0)

    # Base growth: stronger while overflow is high (cells still spread out),
    # gentler as the layout legalizes so HPWL can be fine-tuned.
    # mu in roughly [1.01, 1.10].
    base = 1.01 + 0.09 * of

    # Trend term: if overflow is *not* improving, push harder; if it is
    # dropping fast, ease off to protect wirelength.
    trend = 0.0
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 2:
        prev = _finite(overflow_history[-2], of)
        last = _finite(overflow_history[-1], of)
        delta = prev - last  # positive => improving
        if delta <= 1e-4:          # stalled: accelerate
            trend = 0.03
        elif delta > 1e-2:         # improving quickly: decelerate
            trend = -0.02

    mu = base + trend

    # Late-stage / well-converged annealing: once overflow is small, stop
    # growing the penalty aggressively so the optimizer recovers HPWL.
    if of < 0.08:
        mu = min(mu, 1.0 + 0.4 * of)

    # Damp updates when gradients are exploding to keep the solve stable.
    if gnorm > 1e3:
        mu = 1.0 + (mu - 1.0) * 0.5

    mu = min(max(mu, 0.98), 1.10)

    new_lambda = cur * mu
    return float(min(max(new_lambda, LO), HI))