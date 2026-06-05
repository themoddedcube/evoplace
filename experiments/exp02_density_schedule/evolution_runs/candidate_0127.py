def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # --- sanitize inputs so we never propagate inf/nan ---
    cl = current_lambda
    if cl != cl or cl == float("inf") or cl == float("-inf") or cl <= 0.0:
        cl = 1.0
    cl = float(cl)

    ovf = overflow
    if ovf != ovf:
        ovf = 1.0
    ovf = min(max(float(ovf), 0.0), 1.0)

    # --- base multiplicative growth, decaying toward ~1 over iterations ---
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # --- overflow trend from history (rising overflow = layout congesting) ---
    delta = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev and prev != float("inf"):
            delta = ovf - float(prev)
    delta = min(max(delta, -0.05), 0.05)

    # Push density weight harder while many bins are over-dense or overflow is
    # rising; ease off as the placement legalizes so HPWL can fine-tune.
    accel = 1.0 + 0.45 * ovf - 1.5 * delta
    accel = min(max(accel, 0.5), 1.6)

    # Late-stage gentleness: once overflow is low, relax growth toward 1.0
    if ovf < 0.10:
        accel = 1.0 + (accel - 1.0) * (ovf / 0.10)

    mu = base * accel

    new_lambda = cl * mu
    if new_lambda != new_lambda or new_lambda == float("inf") or new_lambda == float("-inf"):
        new_lambda = cl

    return min(max(new_lambda, 0.01), 50.0)