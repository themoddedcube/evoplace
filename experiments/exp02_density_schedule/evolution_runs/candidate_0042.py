def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # DREAMPlace-style multiplicative growth, made overflow-adaptive and
    # hard-clamped so the density weight can never run away to inf.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # Base step decays slowly with iteration (anneal toward gentle growth).
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow trend: accelerate while density is still being resolved,
    # ease off (or hold) once cells have spread out / overflow is dropping.
    if len(overflow_history) >= 2:
        delta = overflow_history[-1] - overflow_history[-2]
    else:
        delta = 0.0

    # High overflow -> push harder; low overflow -> slow down.
    # Rising overflow (delta>0) nudges growth up; falling overflow eases it.
    overflow_factor = 0.5 + min(max(overflow, 0.0), 1.0)
    trend_factor = 1.0 + max(min(delta, 0.05), -0.05)

    mu = base * overflow_factor * trend_factor

    # Damp growth as gradients get large to avoid oscillation late in placement.
    if gradient_norm > 0.0:
        mu = min(mu, UPPER_PCOF + 0.5 / (1.0 + gradient_norm))

    mu = min(max(mu, LOWER_PCOF), 1.10)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))