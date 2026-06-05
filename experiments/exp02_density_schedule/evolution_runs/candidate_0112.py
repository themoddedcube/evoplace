def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base geometric decay of the growth rate (DREAMPlace-style annealing):
    # aggressive density push early, gentler as the layout matures.
    base_mu = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: when many bins are over-dense the cells still
    # overlap heavily, so grow the density penalty faster to spread them out.
    # As overflow falls toward a legal layout, throttle growth to let the
    # wirelength gradient fine-tune positions without overshoot.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Map overflow in [0, 1] -> multiplier coefficient in [LOWER, UPPER].
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Detect stalling overflow: if recent history shows little progress,
    # nudge the penalty up to escape the plateau.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        progress = recent[0] - recent[-1]
        if progress < 1e-3 and of > 0.1:
            pcof *= 1.02

    # Damp updates when gradients are exploding to keep optimization stable.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 0.0:
        damp = 1.0 / (1.0 + 0.05 * max(gn - 1.0, 0.0))
        pcof = 1.0 + (pcof - 1.0) * damp

    mu = pcof * base_mu
    new_lambda = current_lambda * mu

    # Clamp to the required output range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)