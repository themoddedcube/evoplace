def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    TARGET_OVERFLOW = 0.10

    # Sanitize inputs (guard against NaN/inf that would poison lambda -> inf).
    of = overflow
    if not (of == of) or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    # DREAMPlace-style decaying base growth, bounded.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow trend over recent history (rising overflow => push harder).
    delta = 0.0
    if overflow_history and len(overflow_history) >= 2:
        try:
            delta = float(overflow_history[-1]) - float(overflow_history[-2])
        except (TypeError, ValueError):
            delta = 0.0

    if of > TARGET_OVERFLOW:
        # Cells still spread out: grow density penalty, faster if overflow rising.
        boost = 1.0 + max(delta, 0.0) * 2.0
        mu = base_mu * boost
        mu = min(mu, UPPER_PCOF)
    else:
        # Near target density: ease off so lambda doesn't overshoot and
        # destabilize the fine-tuning phase; relax mildly if overflow rebounds.
        frac = of / TARGET_OVERFLOW
        mu = 1.0 + (base_mu - 1.0) * frac
        if delta > 0.0:
            mu = max(mu * (1.0 - min(delta, 0.05)), LOWER_PCOF)
        mu = max(mu, LOWER_PCOF)

    new_lambda = current_lambda * mu

    # Hard safety: never propagate NaN/inf, always stay in legal range.
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))