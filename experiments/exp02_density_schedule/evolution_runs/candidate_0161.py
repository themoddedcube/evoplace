def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LOWER_PCOF = 0.95
    UPPER_PCOF = 1.05

    # --- Overflow-adaptive multiplier (DREAMPlace-style, but data-driven) ---
    # Grow the density weight aggressively while bins are congested, and
    # ease off as the layout legalizes so wirelength can relax into place.
    of = overflow if overflow == overflow else 1.0          # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Base growth proportional to remaining overflow: lots of overflow -> push
    # density hard (near UPPER_PCOF); near-legal -> coast (near 1.0).
    mu = 1.0 + (UPPER_PCOF - 1.0) * (of ** 0.5)

    # --- Trend term: react to whether overflow is actually improving ---
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last:                  # both finite
            delta = last - prev
            if delta > 1e-4:                                # overflow rising: push harder
                mu *= 1.03
            elif delta < -1e-4:                             # improving: relax slightly
                mu *= 0.99

    # --- Stall detection: if overflow plateaus high, bump to escape ---
    if len(overflow_history) >= 4:
        window = overflow_history[-4:]
        if all(w == w for w in window):
            spread = max(window) - min(window)
            if spread < 5e-3 and of > 0.1:
                mu *= 1.02

    # --- Late-stage damping: once nearly legal, stop inflating lambda ---
    if of < 0.08:
        mu = min(mu, 1.005)

    # Clamp the per-step multiplier to the safe DREAMPlace band.
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    new_lambda = current_lambda * mu

    # Hard bounds required by the optimizer.
    if not (new_lambda == new_lambda):                      # NaN -> reset to a sane value
        new_lambda = 1.0
    return min(max(new_lambda, 0.01), 50.0)