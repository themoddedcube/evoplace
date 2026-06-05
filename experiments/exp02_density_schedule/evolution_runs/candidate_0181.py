def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base decaying multiplier (DREAMPlace-style), kept bounded.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push density weight harder while bins are
    # congested, relax once the placement is spreading out cleanly.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Estimate recent overflow trend to detect stalls vs. healthy progress.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-1]
        prev = overflow_history[-min(5, len(overflow_history))]
        if recent == recent and prev == prev:
            trend = prev - recent  # >0 means overflow is decreasing (good)

    if of > 0.85:
        # Heavily congested: accelerate density weight growth.
        mu = base_mu * (1.0 + 0.10 * (of - 0.85) / 0.15)
    elif of > 0.10:
        # Mid regime: standard growth, but if overflow stalls, push harder
        # to break the plateau; if dropping fast, ease off to protect HPWL.
        if trend < 1e-4:
            mu = base_mu * 1.03
        else:
            mu = base_mu * (1.0 - 0.20 * min(trend / max(of, 1e-3), 1.0))
        mu = max(mu, LOWER_PCOF)
    else:
        # Nearly legal: stop inflating density weight, hold steady so the
        # wirelength term can fine-tune without overshoot.
        mu = min(base_mu, 1.0)

    # Dampen if gradients are exploding (noisy regime).
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))