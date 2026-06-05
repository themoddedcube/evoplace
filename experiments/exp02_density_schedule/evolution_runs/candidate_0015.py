def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight growth with hard clamping.

    Grows the density penalty geometrically (DREAMPlace style) but
    modulates the multiplier by overflow level and recent overflow
    progress: push harder while bins stay congested, ease off once
    overflow is clearing so HPWL is not over-penalized late. Output is
    clamped to a safe range to prevent the runaway -> inf blow-up of the
    unbounded baseline.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    # --- sanitize inputs (guard against NaN / inf) ---
    of = overflow if overflow == overflow else 1.0
    if of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    cl = current_lambda if current_lambda == current_lambda else 0.01
    if cl in (float("inf"), float("-inf")):
        cl = 50.0
    cl = min(max(cl, 0.01), 50.0)

    # --- baseline geometric growth, decaying with iteration ---
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # --- recent overflow trend: positive delta => improving ---
    delta = 0.0
    if overflow_history and len(overflow_history) >= 1:
        prev = overflow_history[-1]
        if prev == prev and prev not in (float("inf"), float("-inf")):
            delta = min(max(prev, 0.0), 1.0) - of

    # --- adaptive multiplier ---
    if delta <= 0.0:
        # stalled or worsening: lean on overflow magnitude, push harder
        adapt_mu = UPPER_PCOF * (1.0 + 0.5 * of)
    else:
        # improving: relax growth, scaled by remaining overflow
        adapt_mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    mu = 0.5 * base_mu + 0.5 * adapt_mu

    # near-convergence: stop inflating lambda so fine HPWL tuning dominates
    if of < 0.10:
        mu = min(mu, 1.0 + 0.2 * (of / 0.10))

    mu = min(max(mu, 1.0), 1.10)

    new_lambda = cl * mu
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cl

    return float(min(max(new_lambda, 0.01), 50.0))