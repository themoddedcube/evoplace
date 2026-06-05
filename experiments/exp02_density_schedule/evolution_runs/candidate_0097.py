def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # DREAMPlace-style geometric growth of the density penalty, with the
    # growth rate annealing toward a small floor as iterations accumulate.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Sanitize overflow into [0, 1].
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # Overflow-adaptive ramp: push lambda harder while bins are badly
    # over-dense, and ease the growth as the layout approaches legality so
    # the late, low-noise phase can fine-tune HPWL without over-penalizing.
    overflow_factor = 0.55 + 0.85 * of  # in [0.55, 1.40]

    # Plateau breaking: if recent overflow has stalled, give lambda an
    # extra nudge to escape the stagnant region.
    if len(overflow_history) >= 5:
        recent = overflow_history[-5:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-3:
            overflow_factor *= 1.12

    # Gradient guard: when gradients explode, damp the growth slightly to
    # keep the optimization stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 10.0:
            overflow_factor *= 0.95

    mu = 1.0 + (base_mu - 1.0) * overflow_factor
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF * 1.2)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))