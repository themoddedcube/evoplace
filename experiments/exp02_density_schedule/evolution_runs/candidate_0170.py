def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with plateau acceleration."""
    # Base multiplicative growth (DREAMPlace-style), gently annealed.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01
    base_mu = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive blend: high overflow -> push density harder (larger mu);
    # low overflow -> ease off so wirelength can settle.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of
    mu = pcof * base_mu

    # Plateau detection: if overflow stopped improving, accelerate the ramp
    # to escape the stall; if it is dropping fast, relax to refine HPWL.
    if isinstance(overflow_history, list) and len(overflow_history) >= 3:
        recent = [h for h in overflow_history[-3:]
                  if isinstance(h, (int, float)) and h == h]
        if len(recent) >= 2:
            delta = recent[0] - recent[-1]          # positive = improving
            if delta < 1e-3 and of > 0.1:
                mu *= 1.03                           # stalled & still congested
            elif delta > 0.02:
                mu *= 0.99                           # improving quickly, ease up

    # Gradient safeguard: if gradients blow up, damp the update.
    if isinstance(gradient_norm, (int, float)) and gradient_norm == gradient_norm:
        if gradient_norm > 1e3:
            mu = min(mu, 1.0)

    cl = current_lambda if (isinstance(current_lambda, (int, float))
                            and current_lambda == current_lambda
                            and current_lambda > 0.0) else 1.0
    new_lambda = cl * mu

    # Hard clamp to the legal range.
    if new_lambda != new_lambda:                     # NaN -> safe default
        new_lambda = 1.0
    return float(min(max(new_lambda, 0.01), 50.0))