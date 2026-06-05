def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Density-weight (lambda) schedule for DREAMPlace-style global placement.
    # Strategy: multiplicative growth (like the nesterov density ramp), but the
    # growth rate is modulated by how fast overflow is actually falling and is
    # hard-clamped to the legal range so the weight never blows up to inf.

    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base growth that decays toward 1.0 as iterations accumulate, so early
    # iterations push density hard and late iterations stop inflating lambda
    # (letting the low-overflow regime fine-tune HPWL instead).
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: if overflow is stalling (not improving), grow
    # faster to break out of the spread-out state; if overflow is dropping
    # quickly, ease off so we don't over-penalize density and hurt wirelength.
    mu = base
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        delta = prev - overflow  # positive => overflow improving
        rate = delta / (prev + 1e-8)
        if rate < 0.005:
            # Overflow stalled: accelerate density penalty.
            mu = base * (1.0 + 0.5 * (0.005 - rate) / 0.005)
        else:
            # Overflow improving well: relax growth proportionally.
            mu = max(LOWER_PCOF, base * (1.0 - 0.3 * min(rate, 1.0)))

    # Once placement is essentially legal, freeze/relax the weight so the final
    # phase optimizes pure HPWL rather than continuing to inflate density.
    if overflow < 0.1:
        mu = min(mu, LOWER_PCOF)

    # Guard against degenerate gradients producing runaway updates.
    if not (gradient_norm == gradient_norm):  # NaN check
        mu = LOWER_PCOF

    new_lambda = current_lambda * mu

    # Hard clamp to the legal range; also catch inf/NaN.
    if not (new_lambda == new_lambda) or new_lambda == float("inf"):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))