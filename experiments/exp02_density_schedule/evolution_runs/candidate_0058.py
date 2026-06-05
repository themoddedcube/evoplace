def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ ... """
    # DREAMPlace-style multiplicative density-penalty growth, but
    # overflow-adaptive and gradient-normalized to avoid divergence.

    # Base growth multiplier: subunity decay of the boost, as in the
    # original (UPPER_PCOF * 0.9999**iter), but bounded more tightly.
    UPPER_PCOF = 1.05
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push hard while bins are congested,
    # ease off (mu -> ~1) as the layout legalizes so we stop inflating
    # lambda once cells have spread out.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow: if overflow is still dropping, keep pressure;
    # if it stalls or rises, accelerate slightly.
    trend = 0.0
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        trend = recent[-1] - recent[0]                       # >0 means rising

    # Map overflow to a growth factor in [~1.0, ~1.06].
    of_factor = 1.0 + 0.06 * of
    if trend > 0.0:
        of_factor *= 1.02                                    # stalled/rising -> push

    mu = base_mu * of_factor

    # Gradient-norm guard: if gradients blow up, damp growth to keep the
    # optimization from diverging (this is what sends HPWL to inf).
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        mu = min(mu, 1.01)
    elif gn > 1e2:
        mu = min(mu, 1.03)

    # Late-stage fine-tuning: once nearly legal, freeze/relax lambda so
    # the WL term can refine the placement.
    if of < 0.10:
        mu = min(mu, 1.0)

    # Keep the per-step multiplier sane.
    mu = min(max(mu, 0.97), 1.08)

    new_lambda = current_lambda * mu

    # Hard clamp to the required output range.
    if new_lambda != new_lambda:                             # NaN -> safe default
        new_lambda = 1.0
    return float(min(max(new_lambda, 0.01), 50.0))