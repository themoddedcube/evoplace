def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # DREAMPlace-style geometric growth that decays with iteration count.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: grow the density weight aggressively while cells are
    # still spread out (high overflow), but ease off as overflow drops so we
    # stop over-penalizing density and blowing up wirelength near convergence.
    of = overflow if overflow == overflow else 1.0  # NaN guard
    of = min(max(of, 0.0), 1.0)
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * of

    # Stagnation guard: if overflow has stopped improving over the last few
    # iterations, restore full growth to push past the plateau.
    if len(overflow_history) >= 5:
        recent = overflow_history[-5:]
        if recent[-1] >= recent[0] - 1e-4:
            mu = max(mu, base_mu)

    # Divergence guard: damp growth when gradients explode (the cause of inf).
    if gradient_norm == gradient_norm and gradient_norm > 1.0e3:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:  # NaN -> reset to a safe mid value
        new_lambda = 1.0
    return float(min(max(new_lambda, 0.01), 50.0))