def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative lambda schedule with stagnation control."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base decaying multiplier (DREAMPlace-style), bounded growth that
    # tapers as iterations proceed so lambda doesn't blow up late.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: when density overflow is still high the
    # density force must keep ramping; when overflow is low we ease off so
    # the wirelength term can refine the placement.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Map overflow into a multiplier in [LOWER_PCOF, UPPER_PCOF].
    # High overflow -> push harder; low overflow -> relax.
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Detect stagnation in overflow history: if overflow stopped improving,
    # nudge lambda up to escape the plateau.
    if overflow_history is not None and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-3 and of > 0.1:
            mu *= 1.02  # gentle boost when density progress stalls

    # Blend the iteration-decay base with the overflow-adaptive term.
    mu = 0.5 * base_mu + 0.5 * mu

    # Gradient safety: if gradients explode, damp the update to stay stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.0)

    # Guard current_lambda against NaN/inf before updating.
    cl = current_lambda
    if not (cl == cl) or cl in (float("inf"), float("-inf")):
        cl = 1.0

    new_lambda = cl * mu

    # Hard clamp to the required output range.
    if not (new_lambda == new_lambda):  # NaN
        new_lambda = 1.0
    return float(min(max(new_lambda, 0.01), 50.0))