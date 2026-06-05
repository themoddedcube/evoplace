def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight (lambda) multiplier with stall detection."""
    LOWER_PCOF = 0.95
    UPPER_PCOF = 1.05

    # Sanitize inputs so a bad signal can never blow lambda up to inf.
    of = overflow if (overflow == overflow and overflow != float("inf")) else 1.0
    of = min(max(of, 0.0), 1.0)
    cl = current_lambda if (current_lambda == current_lambda
                            and current_lambda not in (float("inf"), float("-inf"))) else 1.0

    # Base growth: strong early to cluster cells, decaying toward 1.0 late.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: high overflow -> push density harder,
    # low overflow (cells well spread) -> ease off so HPWL can be refined.
    of_factor = 0.5 + of  # ranges ~[0.5, 1.5]
    mu = 1.0 + (base - 1.0) * of_factor

    # Stall detection: if overflow has stopped improving over recent history,
    # boost the multiplier to break out of the plateau.
    if overflow_history and len(overflow_history) >= 4:
        recent = [h for h in overflow_history[-4:]
                  if h == h and h not in (float("inf"), float("-inf"))]
        if len(recent) >= 4:
            improvement = recent[0] - recent[-1]
            if improvement < 1e-3 and of > 0.1:
                mu *= 1.08  # stuck and still over-dense -> push harder
            elif improvement < 0 and of < 0.1:
                mu *= LOWER_PCOF  # over-spreading while nearly converged -> back off

    # Late-stage damping: once well converged, stop growing lambda.
    if of < 0.05:
        mu = min(mu, 1.0)

    # Clamp the multiplier to a sane per-step range.
    mu = min(max(mu, 0.9), 1.12)

    new_lambda = cl * mu
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cl

    return float(min(max(new_lambda, 0.01), 50.0))