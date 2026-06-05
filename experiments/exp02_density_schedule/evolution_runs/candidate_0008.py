def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty growth with hard clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Sanitize inputs (NaN/inf guards; NaN != NaN).
    of = overflow if overflow == overflow else 1.0
    if of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    cl = current_lambda if current_lambda == current_lambda else 0.01
    if cl in (float("inf"), float("-inf")):
        cl = 50.0

    # Base geometric growth that decays toward ~1 as iterations advance,
    # matching DREAMPlace's annealed density-weight ramp.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Scale the multiplier by overflow: push hard while many bins are
    # over-dense (cells still spreading), ease toward ~1 as the layout
    # legalizes so wirelength is not over-penalized in fine-tuning.
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * of

    # Stall detection: if overflow stops improving while still high,
    # nudge the penalty harder to break out of the plateau.
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev and of > 0.1 and of >= prev - 1e-4:
            mu *= 1.02

    # Keep the multiplier monotone-increasing but bounded.
    mu = min(max(mu, 1.0), UPPER_PCOF)

    new_lambda = cl * mu
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cl

    return float(min(max(new_lambda, 0.01), 50.0))