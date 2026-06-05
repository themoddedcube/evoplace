def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # --- Overflow-adaptive density-weight (lambda) schedule ---
    # Push lambda up while cells are still spread out (high overflow),
    # ease off and fine-tune once the layout is nearly legal (low overflow).

    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)

    # Base growth, DREAMPlace-style, with a gentle decay of aggressiveness.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.0
    base_mu = max(UPPER_PCOF * (0.9999 ** float(iteration)), 0.99)

    # Overflow feedback: ramp harder when far from a legal layout,
    # shrink toward 1.0 (hold) as overflow approaches the target.
    TARGET_OF = 0.07
    if of > TARGET_OF:
        # Scale extra push by how far above target we are (0..~1).
        excess = (of - TARGET_OF) / (1.0 - TARGET_OF)
        mu = 1.0 + (base_mu - 1.0) * (0.5 + excess)
    else:
        # Close to legal: stop growing, allow slight relaxation for HPWL.
        deficit = (TARGET_OF - of) / TARGET_OF
        mu = max(LOWER_PCOF - 0.02 * deficit, 0.97)

    # Trend feedback: if overflow is rising (placement loosening), push more.
    if isinstance(overflow_history, list) and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] > recent[0] + 1e-4:
            mu *= 1.02
        elif recent[-1] < recent[0] - 1e-4:
            mu *= 0.995  # converging nicely, ease the ramp

    # Gradient feedback: damp growth when gradients explode to avoid divergence.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        mu = min(mu, 1.01)

    # Clamp per-step multiplier to a safe band.
    mu = min(max(mu, 0.95), 1.08)

    new_lambda = current_lambda * mu

    # Final hard clamp on the returned value.
    return float(min(max(new_lambda, 0.01), 50.0))