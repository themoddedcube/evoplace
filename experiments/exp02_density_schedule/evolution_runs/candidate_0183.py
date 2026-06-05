def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric cooling of the growth rate (matches DREAMPlace intent:
    # push density hard early, ease off as iterations accumulate).
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive multiplier: when many bins are still over-dense we want
    # lambda to keep climbing; once overflow collapses we slow growth so we do
    # not over-penalize density at the expense of HPWL accuracy.
    ofl = overflow if (overflow == overflow) else 1.0  # guard against NaN
    ofl = min(max(ofl, 0.0), 1.0)

    # Trend of overflow over recent history; if it is stalling, nudge harder.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-1]
        if prev == prev:  # not NaN
            trend = ofl - min(max(prev, 0.0), 1.0)

    # Map overflow + stall trend into a bounded growth factor.
    # High overflow  -> mu near UPPER_PCOF (grow lambda).
    # Low overflow   -> mu near 1.0        (hold lambda, refine wirelength).
    base = 1.0 + (UPPER_PCOF - 1.0) * ofl
    if trend >= 0.0:                       # overflow not improving: encourage growth
        base += (UPPER_PCOF - 1.0) * 0.5 * trend
    else:                                  # overflow falling nicely: relax growth
        base += (base - 1.0) * 0.5 * (trend * 5.0)

    mu = base * decay
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)  # keep step safe and bounded

    new_lambda = current_lambda * mu

    # Hard clamp to the legal range; also sanitize non-finite values.
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))