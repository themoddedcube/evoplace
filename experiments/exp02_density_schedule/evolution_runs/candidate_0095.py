def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty (lambda) schedule.

    Grows lambda quickly while density overflow is high (cells still
    spread out), then anneals the growth rate as overflow drops so the
    optimizer can fine-tune HPWL without exploding the penalty term.
    """
    # --- safe local copies / fallbacks -------------------------------
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)
    lam = current_lambda if current_lambda == current_lambda else 1.0
    if lam <= 0.0:
        lam = 1.0

    # --- overflow trend (are we still making progress?) --------------
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        cur = overflow_history[-1]
        if prev == prev and cur == cur:
            trend = prev - cur            # >0 means overflow decreasing

    # --- base multiplicative growth (DREAMPlace style) ---------------
    # Decay the upper bound on growth with iteration so late-stage steps
    # are gentle, mirroring high-gamma-early -> low-late intuition.
    upper = 1.05
    decay = max(0.9999 ** float(iteration), 0.98)
    base_mu = upper * decay

    # --- overflow-adaptive scaling -----------------------------------
    # High overflow  -> push lambda harder to compact the layout.
    # Low overflow   -> ease off so HPWL gradients dominate.
    of_boost = 1.0 + 0.30 * of
    mu = 1.0 + (base_mu - 1.0) * of_boost

    # If overflow has stalled (barely decreasing) while still high,
    # add a small extra kick to escape the plateau.
    if of > 0.20 and 0.0 <= trend < 0.005:
        mu *= 1.02

    # If overflow is essentially resolved, stop growing the penalty and
    # let it relax slightly to recover wirelength.
    if of < 0.05:
        mu = min(mu, 1.0)

    # --- gradient-norm safety brake ----------------------------------
    # Runaway gradients signal instability; damp the growth.
    if gradient_norm == gradient_norm and gradient_norm > 1e4:
        mu = min(mu, 1.0)

    # --- clamp growth factor and apply -------------------------------
    mu = min(max(mu, 0.95), 1.10)
    new_lambda = lam * mu

    # --- final hard clamp to legal range -----------------------------
    if new_lambda != new_lambda:        # NaN guard
        new_lambda = lam
    return float(min(max(new_lambda, 0.01), 50.0))