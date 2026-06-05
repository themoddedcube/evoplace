def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule.

    Grows lambda to spread cells while overflow is high, then anneals the
    growth as the layout legalizes so HPWL can be fine-tuned without the
    density penalty overshooting (which sends HPWL to inf).
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Classic DREAMPlace decaying ceiling on the per-step growth.
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive factor: clamp overflow into [0, 1] defensively.
    ovfl = overflow if overflow == overflow else 1.0  # NaN guard
    ovfl = min(max(ovfl, 0.0), 1.0)

    # Detect the trend in overflow to decide whether to push or ease off.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-min(5, len(overflow_history)):]
        trend = recent[-1] - recent[0]

    # Map overflow to a multiplier in [LOWER_PCOF, UPPER_PCOF].
    # High overflow -> push lambda up; low overflow -> let it settle.
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (ovfl ** 0.5)
    mu *= base

    # Stagnation / oscillation handling: if overflow stopped improving while
    # still high, nudge harder; if it is rising (diverging), ease off.
    if trend > 1e-4 and ovfl > 0.1:
        mu *= 0.97          # overflow growing -> damp to avoid blow-up
    elif abs(trend) < 1e-4 and ovfl > 0.1:
        mu *= 1.02          # stuck but not legal -> escalate

    # Gradient safety: very large gradients mean we are far from converged;
    # avoid amplifying lambda into instability.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.0)

    # Once nearly legal, freeze growth for accurate HPWL fine-tuning.
    if ovfl < 0.07:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Hard clamp to the required range.
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))