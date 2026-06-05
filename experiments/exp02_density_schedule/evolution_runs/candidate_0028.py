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

    # overflow-trend control: react to how fast overflow is dropping so we
    # neither stall (too slow -> push harder) nor overshoot (too fast -> ease)
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent            # positive == overflow decreasing
        if delta <= 0.0:
            coef *= 1.06                  # stalled/rising: increase pressure
        elif delta <= 5e-5:
            coef *= 1.05
        elif delta <= 5e-4:
            coef *= 1.025
        elif delta > 1.5e-2:
            coef *= 0.955                 # collapsing too fast: ease off
        elif delta > 8e-3:
            coef *= 0.98
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # near-legal regime: relax density pressure so cells re-cluster and
    # wirelength tightens, letting overflow ride up toward (but below) the
    # legality target instead of over-spreading and leaving HPWL on the table
    if of < 0.06:
        coef *= 0.88 + 1.0 * of
    elif of < 0.10:
        coef *= 0.94
    elif of < 0.18:
        coef *= 0.985

    # gradient safeguard: throttle updates when gradients blow up to keep
    # the optimization numerically stable
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    # gate-safety guard (final override): as overflow approaches the legality
    # ceiling, force net-increasing density pressure so the endgame can ride
    # close to the target without ever drifting into the illegal band
    if of > 0.105:
        coef = max(coef, 1.02)
    if of > 0.115:
        coef = max(coef, 1.05)

    mu = coef * base
    mu = min(max(mu, 0.90), 1.10)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))