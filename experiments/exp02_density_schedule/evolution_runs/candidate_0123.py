def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty schedule (RePlAce-style μ with feedback)."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base RePlAce-like ramp: aggressive early, gently decaying.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # --- Overflow feedback: scale penalty growth by how legal we already are. ---
    # High overflow -> push harder; low overflow -> ease off toward fine-tuning.
    ov = overflow if overflow == overflow else 1.0  # guard NaN
    ov = min(max(ov, 0.0), 1.0)

    if ov > 0.9:
        mu = base * 1.06          # cells still badly clustered: spread faster
    elif ov > 0.5:
        mu = base                 # healthy mid-phase: nominal RePlAce ramp
    elif ov > 0.1:
        mu = LOWER_PCOF + 0.10 * (ov - 0.1) / 0.4   # ramp down as we legalize
    else:
        mu = LOWER_PCOF           # nearly legal: relax penalty for HPWL polish

    # --- Stagnation / oscillation detection from overflow trajectory. ---
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[-1] - recent[0]
        # Overflow rising again (cells re-spreading): boost penalty.
        if delta > 0.01:
            mu *= 1.03
        # Overflow plateaued while still illegal: nudge harder to break stall.
        elif abs(delta) < 0.002 and ov > 0.1:
            mu *= 1.02

    # --- Gradient-norm guard: avoid blowing up when gradients are exploding. ---
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.0)     # don't amplify an already-large force field

    new_lambda = current_lambda * mu

    # Clamp to the legal range.
    if new_lambda != new_lambda:      # NaN fallback
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))