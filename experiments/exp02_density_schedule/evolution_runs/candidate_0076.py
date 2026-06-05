def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    # DREAMPlace-style multiplicative density-weight update, made
    # overflow-adaptive so the penalty ramps fast while cells are badly
    # spread out and gently as the layout legalizes.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base geometric step that decays toward 1.0 as iterations grow,
    # so early iterations push hard and late iterations fine-tune.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive blend: high overflow -> push the upper multiplier,
    # low overflow -> relax toward the lower multiplier.
    of = overflow if overflow == overflow else 1.0   # guard NaN
    of = min(max(of, 0.0), 1.0)
    mu = LOWER_PCOF + (base - LOWER_PCOF) * of

    # Trend damping: if overflow is already falling, ease off the ramp to
    # avoid overshooting density and corrupting HPWL late in placement.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] < recent[0]:
            drop = min(max(recent[0] - recent[-1], 0.0), 0.1)
            mu = 1.0 + (mu - 1.0) * (1.0 - 5.0 * drop)

    # Gradient safeguard: if gradients explode, hold lambda steady rather
    # than amplifying the instability.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.0 + (mu - 1.0) * 0.25)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))