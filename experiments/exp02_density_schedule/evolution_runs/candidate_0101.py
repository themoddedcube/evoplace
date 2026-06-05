def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ ... """
    UPPER_PCOF = 1.05

    # Proven backbone: DREAMPlace's improving-HPWL multiplicative ramp.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Recent overflow trend (negative == cells spreading / converging).
    if len(overflow_history) >= 3:
        delta = overflow_history[-1] - overflow_history[-3]
    elif len(overflow_history) >= 2:
        delta = overflow_history[-1] - overflow_history[-2]
    else:
        delta = 0.0

    if delta >= -1e-4:
        # Stalled or worsening overflow: density penalty isn't biting yet,
        # so accelerate the ramp to break the plateau.
        mu = base_mu * 1.05
    else:
        # Converging: ease off the ramp in proportion to spreading progress
        # so we don't over-penalize density and distort HPWL gradients.
        progress = min(-delta / max(overflow, 1e-6), 1.0)
        mu = base_mu * (1.0 - 0.15 * progress)

    # Phase-based fine-tuning: as placement nears legality, freeze/relax the
    # density weight so the optimizer sharpens HPWL on accurate gradients.
    if overflow < 0.08:
        mu = min(mu, 1.0)
    elif overflow < 0.18:
        mu = min(mu, base_mu)

    # Safety: guard against a runaway gradient blowing lambda up in one step.
    if gradient_norm > 0.0 and current_lambda > 0.0:
        mu = min(mu, 1.10)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))