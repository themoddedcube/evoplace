def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Sanitize inputs to avoid inf/nan propagation that blows the schedule up.
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    cur = _finite(current_lambda, 1.0)
    ovf = _finite(overflow, 1.0)
    gnorm = _finite(gradient_norm, 1.0)
    if cur <= 0.0:
        cur = 0.01

    # Base geometric growth on the density weight (DREAMPlace-style increasing lambda).
    # Slightly stronger than baseline early, decaying toward a gentle floor late.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: when overflow is high, cells are still poorly spread,
    # so push the density weight harder; when overflow is low, ease off so HPWL
    # (wirelength) can be fine-tuned without over-spreading.
    ovf = min(max(ovf, 0.0), 1.0)
    overflow_gain = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * ovf  # in [0.95, 1.05]

    # Stagnation detection: if overflow has barely moved over recent history,
    # give an extra nudge to escape the plateau.
    stagnation = 1.0
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 4:
        recent = [_finite(v, ovf) for v in overflow_history[-4:]]
        progress = recent[0] - recent[-1]
        if progress < 1e-3:
            stagnation = 1.02

    # Gradient damping: if gradients are exploding, grow more conservatively.
    grad_damp = 1.0
    if gnorm > 0.0 and gnorm > 1e3:
        grad_damp = 0.98

    mu = base_mu * overflow_gain * stagnation * grad_damp

    next_lambda = cur * mu
    if next_lambda != next_lambda or next_lambda in (float("inf"), float("-inf")):
        next_lambda = cur

    # Hard clamp to the allowed range (prevents the inf blow-up seen in baseline).
    return float(min(max(next_lambda, 0.01), 50.0))