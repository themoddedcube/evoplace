def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # sanitize overflow
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # mild global decay floor: density pressure relaxes very slowly as
    # placement matures so late iterations can fine-tune wirelength
    base = max(0.99985 ** float(iteration), 0.98)

    # overflow-magnitude coefficient (DREAMPlace-style): strong push to
    # spread cells while overflow is high, gentle once nearly legal
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.85)

    # maturity gate in [0,1]: 0 early (spreading must proceed), 1 late.
    maturity = min(max((float(iteration) - 120.0) / 260.0, 0.0), 1.0)
    # overflow gate in [0,1]: ~0 once overflow has settled into the healthy
    # low-HPWL plateau (~0.35), ~1 while cells are still badly clustered.
    of_gate = min(max((of - 0.30) / 0.35, 0.0), 1.0)

    # overflow-trend control: smooth, continuous response to how fast
    # overflow is dropping. A saturating map of the decrease rate reacts
    # proportionally -- push harder when stalled, ease when collapsing --
    # without discontinuous jumps that let one noisy step overshoot.
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent            # positive == overflow decreasing
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))          # in (-1, 1)
        if sat <= 0.0:
            # stall-boost: ONLY meaningful while there is still real
            # spreading left to do. A natural plateau at the low-HPWL
            # endpoint reads as "stalled" too; boosting there just
            # over-spreads cells and inflates HPWL. Gate the boost down as
            # placement matures and as overflow settles, so we stop fighting
            # a healthy plateau.
            stall_strength = 0.058 * (0.25 + 0.75 * of_gate) * (1.0 - 0.55 * maturity)
            coef *= 1.0 - stall_strength * sat
        else:
            # easing off when overflow is collapsing is always safe -- keep it
            coef *= 1.0 - 0.046 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.0 + 0.03 * (0.25 + 0.75 * of_gate)

    # near-legal regime: damp lambda growth so wirelength can be refined
    # without re-spreading already-placed cells
    if of < 0.06:
        coef *= 0.88 + 1.0 * of
    elif of < 0.10:
        coef *= 0.95
    elif of < 0.18:
        coef *= 0.985

    # gradient safeguard: throttle updates when gradients blow up
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    mu = coef * base

    # iteration-aware growth ceiling: cap how fast lambda can still rise late
    # in the schedule. Once the layout is mostly formed an aggressive density
    # push only over-spreads cells (the failure mode that collapses overflow
    # toward ~0.08 with badly inflated HPWL), so tighten the upper bound on mu
    # over time -- slightly more than before -- while leaving early spreading
    # unconstrained.
    hi = 1.10 - 0.07 * min(max((float(iteration) - 220.0) / 260.0, 0.0), 1.0)
    mu = min(max(mu, 0.90), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))