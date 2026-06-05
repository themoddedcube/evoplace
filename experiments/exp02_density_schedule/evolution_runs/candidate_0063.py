def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # --- Overflow-adaptive density-weight growth (DREAMPlace-style mu) ---
    # lambda is the density penalty multiplier; it must GROW over time to
    # drive cells apart, but the growth rate should track how congested the
    # layout still is (high overflow -> push harder, low overflow -> ease off
    # so the wirelength term can fine-tune and HPWL converges accurately).

    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)

    # Base geometric schedule (the original, proven anchor).
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive gain: interpolate the multiplier between a gentle
    # lower bound (near convergence) and the aggressive upper bound (still
    # congested). overflow ~ 1.0 -> use UPPER_PCOF; overflow ~ target -> LOWER.
    TARGET_OF = 0.10
    span = max(of - TARGET_OF, 0.0) / max(1.0 - TARGET_OF, 1e-6)
    span = min(max(span, 0.0), 1.0)
    mu_of = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * span

    # Detect stagnation in overflow: if overflow has stopped improving,
    # nudge lambda harder to break out of the plateau.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        prev = recent[0]
        last = recent[-1]
        if prev == prev and last == last:
            improvement = prev - last
            if improvement < 1e-3 and of > TARGET_OF:
                mu_of = min(mu_of * 1.02, 1.10)   # stalled & congested -> push

    # Blend the time-decayed base with the overflow-adaptive multiplier.
    mu = 0.5 * base + 0.5 * mu_of

    # Late-stage annealing: once overflow is essentially resolved, stop
    # inflating lambda so the optimizer can settle on accurate HPWL.
    if of <= TARGET_OF:
        mu = min(mu, 1.0)

    # Gradient-norm safety: if gradients are exploding, damp growth to avoid
    # destabilizing the placement; if vanishing, allow a touch more push.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = 0.5 * mu + 0.5 * 1.0

    new_lambda = current_lambda * mu

    # Clamp to the legal range.
    if new_lambda != new_lambda:                 # NaN guard
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))