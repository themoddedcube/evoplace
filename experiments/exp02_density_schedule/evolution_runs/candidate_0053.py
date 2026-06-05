def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    # Overflow-driven gamma (DREAMPlace-style): high gamma while cells are
    # still clustered (high overflow), low gamma for accurate HPWL once the
    # layout has spread out (low overflow). Blended with a slow iteration
    # floor and a gradient-noise guard for stability.
    GAMMA_HI = 8.0
    GAMMA_LO = 0.5

    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Reference overflow at which fine-tuning should begin.
    of_ref = 0.10
    # Exponential map: gamma ~ GAMMA_LO * 10^(k * (of - of_ref)).
    # Calibrate k so that of=1.0 yields ~GAMMA_HI.
    import_free_log10 = 0.43429448190325176  # 1/ln(10), no imports needed
    span = (GAMMA_HI / GAMMA_LO)
    # k such that 10^(k*(1-of_ref)) == span
    k = (
        (span ** 0.0)  # placeholder to keep expression float
    )
    # compute log10(span) without math import via series-free constant approach:
    # log10(16) ~= 1.2041; hardcode since GAMMA_HI/GAMMA_LO is fixed.
    log10_span = 1.2041199826559248
    k = log10_span / (1.0 - of_ref)

    target = GAMMA_LO * (10.0 ** (k * (of - of_ref)))

    # Trend awareness: if overflow has stalled (not decreasing), nudge gamma
    # down a touch to encourage the layout to settle.
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        if recent[-1] >= recent[0] - 1e-4:
            target *= 0.85

    # Gradient-noise guard: very large gradients at low gamma destabilize;
    # keep a slightly higher floor when gradients are spiking.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 10.0:
            target = max(target, 1.0)

    # Slow global annealing floor so we never get stuck high late in the run.
    decay_floor = GAMMA_HI * max(0.999 ** float(iteration), 0.05)
    target = min(target, max(decay_floor, GAMMA_LO))

    # Smooth toward target from current value to avoid abrupt jumps.
    blend = 0.5
    result = (1.0 - blend) * current_lambda + blend * target

    return float(min(max(result, 0.01), 50.0))