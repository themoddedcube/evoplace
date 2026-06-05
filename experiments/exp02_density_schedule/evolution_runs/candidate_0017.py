def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    INF = float("inf")

    def clean(x, d):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return d
        if x != x or x == INF or x == -INF:
            return d
        return x

    # --- sanitize inputs ---
    of = min(max(clean(overflow, 1.0), 0.0), 1.0)
    lam = clean(current_lambda, 1.0)
    if lam <= 1e-8:
        lam = 1e-3
    gnorm = max(clean(gradient_norm, 0.0), 0.0)
    it = int(iteration) if iteration == iteration else 0

    # --- overflow trend (negative = spreading/improving) ---
    hist = [min(max(clean(h, of), 0.0), 1.0) for h in (overflow_history or [])]
    if len(hist) >= 4:
        recent = sum(hist[-3:]) / 3.0
        older = sum(hist[-6:-3]) / max(len(hist[-6:-3]), 1)
        trend = recent - older
    else:
        trend = 0.0

    # --- schedule parameters ---
    TARGET = 0.10            # ePlace-style stopping overflow
    UPPER, LOWER = 1.05, 0.96

    # base multiplier: ramp weight up in proportion to overflow excess
    excess = max(of - TARGET, 0.0) / (1.0 - TARGET)   # in [0, 1]
    mu = 1.0 + (UPPER - 1.0) * excess

    if of > TARGET:
        # still too dense: if spreading has stalled, push harder
        if trend >= 0.0:
            mu *= 1.01
    else:
        # overflow met: relax weight so HPWL term can fine-tune
        mu = LOWER + (1.0 - LOWER) * (of / TARGET)

    # gradient safeguard: damp growth when gradients are exploding
    if gnorm > 1e3:
        mu = min(mu, 1.0 + (mu - 1.0) * (1e3 / gnorm))

    # mild warmup so early iterations don't over-accelerate
    if it < 10:
        mu = 1.0 + (mu - 1.0) * (0.5 + 0.05 * it)

    mu = min(max(mu, LOWER), UPPER)

    new_lam = lam * mu
    return float(min(max(new_lam, 0.01), 50.0))