def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Overflow-adaptive multiplicative growth of the density penalty.
    # Standard DREAMPlace ramps lambda by a fixed factor each step; here we
    # modulate that factor by how fast overflow is shrinking, so the penalty
    # pushes harder while cells are still spread out and eases off once the
    # layout is nearly legal (letting low-lambda HPWL gradients fine-tune).

    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Bounded, well-conditioned inputs.
    ov = overflow if overflow == overflow else 1.0   # guard NaN
    ov = min(max(ov, 0.0), 1.0)
    gnorm = gradient_norm if gradient_norm == gradient_norm else 1.0
    cur = current_lambda if current_lambda == current_lambda else 1.0
    cur = min(max(cur, 0.01), 50.0)

    # Trend of overflow over recent history: negative => improving fast.
    delta = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-1]
        prev = overflow_history[-min(len(overflow_history), 5)]
        if recent == recent and prev == prev:
            delta = recent - prev   # >0 means overflow rising (bad)

    # Base ramp that decays toward 1.0 as iterations progress.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Map overflow in [0,1] to a growth factor:
    #  - high overflow -> grow faster (up to UPPER_PCOF * (1 + slack))
    #  - low overflow  -> grow slowly or hold (down toward LOWER_PCOF)
    ov_factor = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (ov ** 0.5)

    # If overflow is creeping back up, push the penalty harder.
    trend_boost = 1.0 + max(delta, 0.0) * 2.0

    mu = base * ov_factor * trend_boost

    # Once nearly legal, relax the penalty so accurate-HPWL gradients dominate.
    if ov < 0.08:
        mu = min(mu, 1.0)

    # Damp explosive multipliers when gradients are already large.
    if gnorm > 0.0:
        mu = mu / (1.0 + 0.05 * max(gnorm - 1.0, 0.0))

    # Keep the per-step multiplier sane.
    mu = min(max(mu, 0.90), 1.15)

    new_lambda = cur * mu
    return float(min(max(new_lambda, 0.01), 50.0))