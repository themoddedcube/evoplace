def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # RePlAce/DREAMPlace-style overflow-adaptive multiplicative density-weight update.
    # Grow lambda fast while spreading is making progress; throttle near convergence;
    # hard-clamp the result so the run can never diverge to inf.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    # Base decay of the growth rate over time (as in the original schedule).
    base = max(0.9999 ** float(iteration), 0.98)

    # Trend of overflow: negative delta == bins are clearing (good progress).
    delta = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        delta = float(overflow_history[-1]) - float(overflow_history[-2])

    # Map overflow trend -> growth coefficient in [LOWER_PCOF, UPPER_PCOF].
    # Strong improvement -> push density harder; stalling/regressing -> ease off.
    if delta < -1e-3:
        pcof = UPPER_PCOF
    elif delta > 1e-3:
        pcof = LOWER_PCOF
    else:
        pcof = 0.5 * (UPPER_PCOF + LOWER_PCOF)

    # As overflow drops, cells are placed legally; stop inflating lambda so the
    # objective can refine HPWL with low penalty distortion.
    of = max(0.0, min(1.0, float(overflow)))
    pcof = 1.0 + (pcof - 1.0) * of

    # Guard against exploding gradients: if the gradient is very large, the
    # penalty term is already dominating, so do not amplify it further.
    if gradient_norm is not None and gradient_norm > 1e6:
        pcof = min(pcof, 1.0 + 0.25 * (pcof - 1.0))

    mu = pcof * base
    new_lambda = current_lambda * mu

    # Enforce the required output range.
    return float(min(max(new_lambda, 0.01), 50.0))