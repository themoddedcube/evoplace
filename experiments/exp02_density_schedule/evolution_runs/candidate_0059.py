def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    # --- sanitize inputs (hook may pass None / NaN / non-finite) ---
    def _f(x, d):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return d
        if x != x or x in (float("inf"), float("-inf")):
            return d
        return x

    it = max(0, int(iteration))
    ovf = min(1.0, max(0.0, _f(overflow, 1.0)))
    lam = _f(current_lambda, 1.0)
    if lam <= 0.0:
        lam = 0.01

    LOWER, UPPER = 0.01, 50.0

    # --- DREAMPlace default improving-HPWL multiplier (baseline ramp) ---
    UPPER_PCOF = 1.05
    base_mu = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # --- overflow-adaptive boost: when bins are still over-dense we want
    #     lambda to climb faster to push cells apart; once overflow is low
    #     we damp the ramp so HPWL can fine-tune without density blowing up.
    #     ovf ~ 1 -> factor ~ 1.04 (accelerate), ovf ~ 0 -> factor ~ 0.985 (ease off)
    spread_factor = 0.985 + 0.055 * ovf

    # --- stagnation detection: if overflow has stopped falling, nudge lambda
    #     up a touch to break the plateau; if it is dropping nicely, hold back.
    stagn = 1.0
    if overflow_history:
        recent = [_f(v, ovf) for v in overflow_history[-5:]]
        if len(recent) >= 2:
            drop = recent[0] - recent[-1]          # positive == improving
            if drop <= 1e-4 and ovf > 0.1:         # plateaued while still dense
                stagn = 1.02
            elif drop > 0.01:                      # converging fast -> relax
                stagn = 0.99

    mu = base_mu * spread_factor * stagn

    # keep per-step growth sane so a single step can't explode lambda
    mu = min(1.10, max(0.90, mu))

    new_lambda = lam * mu

    # gentle decay of the ceiling pressure late in the run for HPWL accuracy
    if ovf < 0.05 and it > 200:
        new_lambda *= 0.995

    return min(UPPER, max(LOWER, new_lambda))