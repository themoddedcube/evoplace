def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Sanitize inputs
    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    cl = current_lambda if current_lambda == current_lambda else 1.0

    # Base multiplicative growth, annealed slightly with iteration so the
    # density penalty ramps hard early (cells still overlapping) and gently
    # later (fine HPWL tuning).
    anneal = max(0.9999 ** float(iteration), 0.985)

    # Overflow-adaptive exponent: large overflow -> push lambda up quickly to
    # spread cells; small overflow -> nearly hold lambda so the wirelength
    # gradient dominates and HPWL is refined.
    #   of = 1.0 -> mu ~ UPPER_PCOF ; of -> 0 -> mu ~ 1.0 (hold)
    mu = 1.0 + (UPPER_PCOF - 1.0) * (of ** 0.5)

    # Stagnation / divergence damping using overflow trend.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-1]
        prev = prev if prev == prev else of
        if of > prev + 1e-4:
            # overflow rising (cells re-clustering / instability) -> ease off
            mu = max(LOWER_PCOF, mu * 0.97)

    # Gradient-norm guard: if gradients blow up, slow the penalty growth to
    # avoid the divergence that produced inf HPWL.
    if gn > 1e3:
        mu = min(mu, 1.0 + (UPPER_PCOF - 1.0) * 0.25)

    mu = min(max(mu * anneal, LOWER_PCOF), UPPER_PCOF)

    new_lambda = cl * mu
    if new_lambda != new_lambda:                            # final NaN guard
        new_lambda = 1.0
    return float(min(max(new_lambda, 0.01), 50.0))