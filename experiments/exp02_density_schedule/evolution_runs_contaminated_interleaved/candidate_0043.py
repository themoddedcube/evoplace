def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # --- sanitize inputs ---
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)
    it = float(iteration)

    # --- base growth envelope: strong early, gently tapering ---
    # decays from ~1.0 toward a 0.985 floor so lambda keeps creeping up
    # but never stalls completely in the long tail.
    base = max(0.99985 ** it, 0.985)

    # --- overflow-adaptive core multiplier ---
    # high overflow -> push lambda up (spread cells);
    # low overflow  -> ease off so HPWL can fine-tune at accurate gamma.
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.80)

    # --- overflow-derivative (trend) control ---
    # measure how fast overflow is dropping; if progress stalls, lean
    # harder on lambda; if it is collapsing fast, relax to avoid overshoot.
    hist = [h for h in overflow_history if h == h]
    if len(hist) >= 4:
        recent = 0.5 * (hist[-1] + hist[-2])
        older = 0.5 * (hist[-3] + hist[-4])
        delta = older - recent                      # >0 means overflow falling
        x = (delta - 1.0e-3) / 4.0e-3
        sat = x / (1.0 + abs(x))                     # squashed to (-1, 1)
        if sat <= 0.0:
            # stalled / rising overflow -> accelerate spreading
            coef *= 1.0 - 0.060 * sat
        else:
            # healthy progress -> mild relaxation
            coef *= 1.0 - 0.040 * sat
    elif len(hist) >= 2:
        delta = hist[-2] - hist[-1]
        if delta <= 1e-4:
            coef *= 1.03

    # --- regime-based shaping on absolute overflow ---
    # below the legalization knee, back off progressively so the final
    # placement settles at accurate (low-gamma) wirelength.
    if of < 0.05:
        coef *= 0.86 + 1.2 * of
    elif of < 0.10:
        coef *= 0.94
    elif of < 0.18:
        coef *= 0.98

    # --- gradient-norm damping: avoid blowing up on noisy steps ---
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 5e4:
            coef *= 0.90
        elif gradient_norm > 1e4:
            coef *= 0.955

    mu = coef * base

    # --- adaptive clamp on the per-step multiplier ---
    # early: allow brisk growth; late: tighten toward 1.0 so lambda
    # stabilizes and the optimizer can converge HPWL.
    prog = min(max((it - 220.0) / 260.0, 0.0), 1.0)
    hi = 1.10 - 0.07 * prog
    lo = 0.90 + 0.04 * prog
    mu = min(max(mu, lo), hi)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))