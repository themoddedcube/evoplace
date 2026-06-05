def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Sanitize inputs so a single NaN/inf can never poison the schedule.
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    it = max(0, int(iteration))
    of = min(1.0, max(0.0, _finite(overflow, 1.0)))
    gnorm = _finite(gradient_norm, 1.0)
    lam = _finite(current_lambda, 1.0)
    if lam <= 0.0:
        lam = 0.01

    # DREAMPlace-style geometric growth of the density weight, but the
    # per-iteration multiplier is modulated by how far placement still is
    # from a legal density (overflow). Far from legal -> push density harder;
    # near-legal -> ease off so wirelength can be fine-tuned.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    # base growth decays slowly with iteration, exactly as the seed did.
    base = max(0.9999 ** float(it), 0.98)

    # Overflow target band: above 0.1 we are still spreading cells, so grow
    # toward UPPER_PCOF; as overflow falls we interpolate toward LOWER_PCOF.
    span = UPPER_PCOF - LOWER_PCOF
    of_factor = LOWER_PCOF + span * min(1.0, of / 0.10)
    mu = of_factor * base

    # Detect overflow stalling/oscillation from history and damp growth so we
    # do not over-penalize density once progress flattens.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 4:
        recent = [_finite(h, of) for h in overflow_history[-4:]]
        improvement = recent[0] - recent[-1]
        if improvement <= 1e-4:          # no longer reducing overflow
            mu = 1.0 + (mu - 1.0) * 0.5  # halve the excess growth
        if recent[-1] > recent[0]:       # overflow rising -> back off
            mu = min(mu, 1.0)

    # Guard against exploding gradients (the usual cause of HPWL = inf):
    # if the gradient magnitude is large, freeze the weight this step.
    if gnorm > 1e4:
        mu = min(mu, 1.0)

    new_lambda = lam * mu

    # Hard clamp to the legal range; final NaN/inf safety net.
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = lam
    return float(min(50.0, max(0.01, new_lambda)))