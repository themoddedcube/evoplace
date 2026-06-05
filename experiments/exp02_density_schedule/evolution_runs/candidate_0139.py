def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Sanitize inputs
    it = max(0, int(iteration))
    ovf = overflow if (overflow is not None and overflow == overflow) else 1.0
    ovf = min(max(ovf, 0.0), 1.0)
    cur = current_lambda if (current_lambda is not None and current_lambda == current_lambda) else 1.0
    if cur <= 0.0:
        cur = 0.01

    # Base ramp: strong early growth, decaying toward 1.0 as iterations progress.
    base = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # Overflow-adaptive boost: when many bins are over-dense, push the density
    # penalty harder; when overflow is low (cells well spread), back off so the
    # wirelength term dominates for fine HPWL tuning.
    # ovf in [0,1] maps growth multiplier in roughly [LOWER_PCOF, base].
    mu = LOWER_PCOF + (base - LOWER_PCOF) * ovf

    # Trend awareness: if overflow is stalling (not decreasing), accelerate.
    if overflow_history and len(overflow_history) >= 3:
        try:
            recent = float(overflow_history[-1])
            prev = float(overflow_history[-3])
            if recent == recent and prev == prev:
                drop = prev - recent
                if drop < 0.005:          # stagnating -> push harder
                    mu *= 1.02
                elif drop > 0.05:          # collapsing fast -> ease off
                    mu *= 0.99
        except (TypeError, ValueError):
            pass

    # Gradient safety: if gradients blow up, dampen lambda growth.
    if gradient_norm is not None and gradient_norm == gradient_norm:
        if gradient_norm > 1e3:
            mu = min(mu, 1.01)

    mu = min(max(mu, 0.99), 1.10)

    new_lambda = cur * mu
    return float(min(max(new_lambda, 0.01), 50.0))