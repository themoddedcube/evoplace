def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ Overflow-adaptive density-weight schedule with anti-divergence guards. """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base DREAMPlace-style geometric growth, annealed by iteration so the
    # multiplier eases toward 1.0 as placement matures (prevents late blow-up).
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive gain: push harder while bins are congested, back off as
    # the layout legalizes. Maps overflow in [0,1] -> gain in [LOWER_PCOF, base_mu].
    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * of

    # Stall detection: if overflow has stopped improving, nudge mu up to escape.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] >= recent[0] - 1e-4 and of > 0.1:
            mu *= 1.02

    # Gradient-norm safety: if gradients explode, damp growth to avoid the
    # inf/NaN divergence that broke the current candidate.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e6:
        mu = min(mu, 1.0)
    elif gn > 1e4:
        mu = min(mu, 1.01)

    new_lambda = current_lambda * mu

    # Hard clamp to the legal range; floor avoids vanishing density force.
    if new_lambda != new_lambda:                            # NaN -> reset low
        new_lambda = 0.01
    return float(min(max(new_lambda, 0.01), 50.0))