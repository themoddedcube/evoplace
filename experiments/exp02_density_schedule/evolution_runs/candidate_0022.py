def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight ramp with bounded, decaying growth."""
    LO, HI = 0.01, 50.0

    # Sanitize inputs (guard against NaN/inf that poison the run).
    cl = current_lambda
    if not (cl == cl) or cl in (float("inf"), float("-inf")):
        cl = 1.0
    cl = min(max(cl, LO), HI)

    of = overflow
    if not (of == of) or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    # Base multiplicative growth that decays with iteration so lambda
    # ramps hard early (cells still spreading) and gently late (fine-tune).
    base = max(0.9999 ** float(iteration), 0.98)
    mu = 1.05 * base

    # Overflow-adaptive scaling: push harder while bins are congested,
    # ease off (mu -> ~1) as overflow approaches the convergence target.
    target = 0.10
    if of > target:
        # Stronger penalty growth when far from a legal layout.
        congestion = (of - target) / (1.0 - target)
        mu *= 1.0 + 0.30 * congestion
    else:
        # Near-legal: stop inflating lambda, let wirelength gradients win.
        mu = 1.0 + 0.02 * (of / target)

    # Damp growth if overflow has stalled (history flat) to avoid runaway.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if max(recent) - min(recent) < 1e-3:
            mu = 1.0 + 0.5 * (mu - 1.0)

    new_lambda = cl * mu

    # Final hard clamp to the required output range.
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cl
    return min(max(new_lambda, LO), HI)