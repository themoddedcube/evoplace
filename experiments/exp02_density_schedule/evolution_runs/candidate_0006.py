def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive gamma: smooth (high) while cells are clustered,
    accurate (low) as density resolves. Bounded and NaN/inf-safe."""
    GAMMA_MIN = 0.5
    GAMMA_MAX = 8.0

    # Robust current overflow (NaN / inf guard -> assume fully clustered).
    of = overflow
    if of != of or of == float('inf') or of == float('-inf'):
        of = 1.0

    # Smooth with recent history to damp gradient-driven overflow noise.
    if overflow_history:
        recent = [h for h in overflow_history[-3:] if h == h]
        if recent:
            of = 0.5 * of + 0.5 * (sum(recent) / len(recent))

    of = min(max(of, 0.0), 1.0)

    # High overflow -> large gamma (smooth gradients); low overflow -> small
    # gamma (accurate HPWL). Geometric interpolation in [GAMMA_MIN, GAMMA_MAX].
    gamma = GAMMA_MIN * (GAMMA_MAX / GAMMA_MIN) ** of

    # Early warmup floor: keep gamma high until cells settle, independent of
    # noisy initial overflow estimates.
    if iteration < 50:
        gamma = max(gamma, GAMMA_MAX * (0.97 ** float(iteration)))

    # Final NaN guard before clamping to the legal range.
    if gamma != gamma:
        gamma = GAMMA_MAX

    return float(min(max(gamma, 0.01), 50.0))