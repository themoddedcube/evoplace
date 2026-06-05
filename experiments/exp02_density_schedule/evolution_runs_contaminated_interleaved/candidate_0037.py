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

    # gentle baseline anneal, floored so early clustering is preserved
    base = max(0.99985 ** float(iteration), 0.98)

    # DREAMPlace-style overflow coefficient: keep gamma high while spread is poor
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.85)

    # ---- progress (overflow-velocity) feedback -------------------------------
    hist = [h for h in overflow_history if h == h]
    stalled = False
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent                      # >0 means overflow shrinking
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))                     # smooth squash in (-1,1)
        if sat <= 0.0:
            # overflow rising/stalled -> hold gamma up a touch to re-cluster
            coef *= 1.0 - 0.058 * sat
            stalled = True
        else:
            # healthy descent -> let gamma fall a bit faster than vanilla
            coef *= 1.0 - 0.052 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03
            stalled = True

    # ---- low-overflow fine-tuning: push gamma toward the accurate regime -----
    # Once cells are well spread, smaller gamma => sharper HPWL approximation,
    # which is where final wirelength is actually recovered.
    if of < 0.04:
        coef *= 0.80 + 1.5 * of                      # strong drop, accurate WL
    elif of < 0.06:
        coef *= 0.85 + 1.0 * of
    elif of < 0.10:
        coef *= 0.94
    elif of < 0.18:
        coef *= 0.985

    # ---- gradient guards: avoid destabilizing big steps ---------------------
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    mu = coef * base

    # ---- adaptive clamp ------------------------------------------------------
    # Upper bound relaxes over iterations; lower bound (max decay per step)
    # is loosened in the late / low-overflow phase so gamma can actually reach
    # the accurate region instead of plateauing at 0.90 per-step.
    hi = 1.10 - 0.05 * min(max((float(iteration) - 250.0) / 250.0, 0.0), 1.0)
    if of < 0.06 and not stalled:
        lo = 0.82                                    # allow faster fine-tune drop
    elif of < 0.12:
        lo = 0.87
    else:
        lo = 0.90
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))