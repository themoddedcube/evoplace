def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.98

    # sanitize overflow (NaN-safe, clamp to [0,1])
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # Target a final overflow just below the legality/overflow gate (0.12).
    # Spreading only as far as legality demands -- rather than the default's
    # ~0.0825 -- leaves cells slightly tighter, lowering HPWL, while keeping a
    # safety margin so a noisy step cannot cross the gate into rejection.
    TARGET = 0.095

    # Slow global decay so late iterations fine-tune instead of re-spreading.
    base = max(0.9999 ** float(iteration), 0.98)

    if of > TARGET:
        # Proportional push: strong while badly clustered, easing toward 1.0 as
        # overflow approaches target. Concave (sqrt) so the approach is gentle.
        err = min((of - TARGET) / (1.0 - TARGET), 1.0)
        coef = 1.0 + (UPPER_PCOF - 1.0) * (err ** 0.5)
    else:
        # At/under target: relax density pressure so the wirelength gradient can
        # pull cells closer (lower HPWL). Ease only gently and never below
        # LOWER_PCOF, so overflow settles at an equilibrium near TARGET instead
        # of rebounding over the gate.
        deficit = (TARGET - of) / TARGET
        coef = 1.0 - (1.0 - LOWER_PCOF) * deficit

    mu = coef * base

    # Trend damping: soften when overflow is collapsing (over-spreading inflates
    # HPWL); firm up through a stall that is still above target.
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 2:
        drop = hist[-2] - hist[-1]            # >0 == overflow decreasing
        if drop > 0.0 and of > TARGET:
            mu = 1.0 + (mu - 1.0) * 0.6
        elif drop <= 1e-4 and of > TARGET:
            mu *= 1.01

    # Divergence guard: a blown-up gradient means the density force is too
    # strong -- cap growth to prevent the runaway lambda that yields inf HPWL.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 5e4:
            mu = min(mu, 1.0)
        elif gradient_norm > 1e4:
            mu = min(mu, 1.02)

    # Bound the per-step multiplier so one noisy step can neither diverge nor
    # collapse the placement.
    mu = min(max(mu, 0.95), UPPER_PCOF)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))