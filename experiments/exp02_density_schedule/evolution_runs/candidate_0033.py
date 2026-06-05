def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # --- sanitize inputs (avoid inf/nan propagation that wrecks the run) ---
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    it = max(0, int(iteration))
    ovf = min(max(_finite(overflow, 1.0), 0.0), 1.0)
    cur = _finite(current_lambda, 1.0)
    if cur <= 0.0:
        cur = 0.01
    gnorm = _finite(gradient_norm, 1.0)

    # --- base multiplicative growth (DREAMPlace-style), decaying with iter ---
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.003
    decay = max(0.9999 ** float(it), 0.985)
    base_mu = UPPER_PCOF * decay

    # --- overflow-adaptive: push hard while spread out, ease off once packed ---
    # high overflow -> grow faster to pull cells into legal density,
    # low overflow  -> grow slowly so HPWL can be fine-tuned without overshoot.
    overflow_gain = 1.0 + 0.6 * (ovf - 0.10)
    overflow_gain = min(max(overflow_gain, 0.5), 1.4)

    # --- trend term: if overflow is stalling/rising, accelerate; if dropping fast, relax ---
    trend = 0.0
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 4:
        recent = [min(max(_finite(h, ovf), 0.0), 1.0) for h in overflow_history[-4:]]
        trend = recent[-1] - recent[0]  # positive => overflow not improving
    trend_gain = 1.0 + min(max(trend, -0.05), 0.05) * 2.0

    # --- gradient guard: if gradients explode, damp growth to stay stable ---
    grad_gain = 1.0
    if gnorm > 1e3:
        grad_gain = 0.9

    mu = base_mu * overflow_gain * trend_gain * grad_gain
    mu = min(max(mu, LOWER_PCOF), 1.25)

    new_lambda = cur * mu

    # --- once nearly legal, freeze growth to lock in low-HPWL solution ---
    if ovf < 0.07:
        new_lambda = min(new_lambda, cur * 1.01)

    # --- hard clamp to required range ---
    if new_lambda != new_lambda:
        new_lambda = cur
    return float(min(max(new_lambda, 0.01), 50.0))