def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight (lambda) schedule with safe clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # --- sanitize inputs (guard NaN / out-of-range) ---
    of = overflow
    if of != of:          # NaN check
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    gn = gradient_norm
    if gn != gn or gn < 0.0:
        gn = 0.0

    # --- base ramp: aggressive early, gentle late (DREAMPlace-style decay) ---
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # --- overflow trend over recent history (positive => legalizing) ---
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        window = overflow_history[-5:] if len(overflow_history) >= 5 else overflow_history[:]
        trend = float(window[0]) - float(window[-1])

    # --- adapt the multiplier ---
    # High overflow + stalled trend: push density harder.
    # Low overflow or fast convergence: ease off so HPWL can refine.
    accel = 1.0
    if of > 0.10 and trend < 0.005:
        accel = 1.08
    elif trend > 0.02:
        accel = 0.99

    # Scale push by how far from legalized we still are (0.5..1.0 band).
    density_push = 0.5 + 0.5 * of

    mu = base * accel * density_push

    # Damp growth if gradients are exploding (numerical safety).
    if gn > 50.0:
        mu = min(mu, 1.0)

    # Keep the per-step multiplier in a stable band.
    mu = min(max(mu, LOWER_PCOF), 1.15)

    next_lambda = current_lambda * mu

    # Final hard clamp to the allowed range.
    return min(max(next_lambda, 0.01), 50.0)