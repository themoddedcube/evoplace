def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # DREAMPlace's default multiplier decay (per-iteration cooling of the ramp).
    base = max(0.9999 ** float(iteration), 0.98)

    # Sanitize overflow into [0, 1] (guard against NaN / out-of-range).
    of = overflow
    if not (of == of):          # NaN check
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    # Overflow-adaptive ramp: push the density weight hard while bins are
    # congested, then ease the multiplier toward 1.0 as overflow approaches
    # the target so the wirelength term can relax HPWL during fine-tuning.
    target = 0.10
    ramp = (of - target) / (1.0 - target)
    ramp = min(max(ramp, 0.0), 1.0)
    ramp = ramp ** 0.5          # hold pressure a little longer as of falls

    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * ramp
    mu = pcof * base

    # Stagnation guard: if overflow has plateaued while still above target,
    # nudge growth to break out of the stalled spreading regime.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if max(recent) - min(recent) < 1e-3 and of > target:
            mu *= 1.01

    new_lambda = current_lambda * mu
    return min(max(new_lambda, 0.01), 50.0)