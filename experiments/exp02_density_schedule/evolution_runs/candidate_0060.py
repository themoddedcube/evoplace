def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-weight schedule.

    Grows the density penalty geometrically while cells are still
    spread out (high overflow), then anneals the growth toward 1.0 as
    overflow collapses so wirelength can be fine-tuned without the
    density term overpowering the HPWL gradient. Includes stall
    detection (boost when overflow plateaus) and gradient-aware
    damping (shrink steps when gradients explode) for robustness.
    """
    # Sanitize inputs (placement can emit nan/inf late in the run).
    it = max(0, int(iteration))
    of = overflow if (overflow == overflow and overflow >= 0.0) else 1.0
    of = min(max(of, 0.0), 1.0)
    cur = current_lambda if (current_lambda == current_lambda and current_lambda > 0.0) else 1.0
    gnorm = gradient_norm if (gradient_norm == gradient_norm and gradient_norm > 0.0) else 0.0

    # Base geometric growth, decaying as iterations progress.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    decay = max(0.9999 ** float(it), 0.985)

    # Overflow shaping: aggressive growth while spread, gentle near convergence.
    # of ~ 1.0 -> near UPPER_PCOF ; of ~ 0.1 -> near LOWER_PCOF.
    target = min(max(of, 0.1), 1.0)
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (target ** 0.7)
    mu *= decay

    # Stall detection: if overflow has plateaued, nudge growth up to escape.
    if isinstance(overflow_history, list) and len(overflow_history) >= 3:
        recent = [h for h in overflow_history[-3:] if h == h]
        if len(recent) >= 3:
            spread = max(recent) - min(recent)
            if spread < 1e-3 and of > 0.1:
                mu *= 1.02

    # Gradient-aware damping: back off if gradients are blowing up.
    if gnorm > 0.0 and len(overflow_history or []) >= 2:
        prev = overflow_history[-1] if overflow_history else gnorm
        if prev == prev and prev > 0.0 and gnorm > 3.0 * prev:
            mu = min(mu, 1.0 + 0.5 * (mu - 1.0))

    # Keep multiplier sane.
    mu = min(max(mu, 0.99), UPPER_PCOF)

    new_lambda = cur * mu
    return float(min(max(new_lambda, 0.01), 50.0))