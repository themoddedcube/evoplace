def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive, bounded density-weight (lambda) update."""
    # --- sanitize inputs (guard NaN/inf without imports) ---
    ovfl = overflow if overflow == overflow else 1.0
    if ovfl == float("inf") or ovfl == float("-inf"):
        ovfl = 1.0
    ovfl = min(max(ovfl, 0.0), 1.0)

    cl = current_lambda if current_lambda == current_lambda else 1.0
    if cl == float("inf") or cl == float("-inf"):
        cl = 1.0
    cl = min(max(cl, 0.01), 50.0)

    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn == float("inf") or gn == float("-inf"):
        gn = 0.0

    # --- base growth scales with remaining overflow ---
    # lots of overlap -> push the density force harder;
    # near-legal -> gentle so HPWL can fine-tune.
    base = 1.0 + 0.055 * ovfl

    # --- trend term from overflow history ---
    if len(overflow_history) >= 2:
        prev = overflow_history[-1]
        if prev == prev and prev != float("inf"):
            prev = min(max(prev, 0.0), 1.0)
            delta = prev - ovfl  # positive => improving
            if delta < 1e-4:     # stalled or rebounding: ease off
                base *= 0.98
            elif delta > 0.01:   # improving well: keep momentum
                base = min(base * 1.01, 1.10)

    # --- gradient guard: damp when gradients blow up ---
    if gn > 1e3:
        base = min(base, 1.01)

    # --- warmup: avoid early overshoot/divergence ---
    if iteration < 5:
        base = min(base, 1.03)

    # --- annealing: once nearly legal, stop growing lambda ---
    if ovfl < 0.08:
        base = min(base, 1.0)

    new_lambda = cl * base
    return float(min(max(new_lambda, 0.01), 50.0))