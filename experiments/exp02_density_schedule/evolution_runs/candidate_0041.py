def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive geometric growth of the density weight with
    stagnation-aware boosting and hard clamping to keep placement stable."""
    # --- Base geometric growth (DREAMPlace-style), but gentler than 1.05 ---
    # A too-aggressive multiplier blows the density force up and HPWL diverges
    # (-> inf). Decay the multiplier toward 1.0 as iterations progress so the
    # late-stage fine-tuning is not destabilized.
    base_mult = 1.03 * max(0.9999 ** float(iteration), 0.985)

    # --- Overflow-adaptive scaling ---
    # High overflow => cells still heavily overlapped => push density harder.
    # Low overflow  => nearly legal => ease off so HPWL can settle.
    of = overflow if overflow == overflow else 1.0  # NaN guard
    of = min(max(of, 0.0), 1.0)
    # Maps overflow in [0,1] to an extra multiplier in ~[0.97, 1.06].
    overflow_factor = 0.97 + 0.09 * of

    # --- Stagnation detection from history ---
    # If overflow has stopped improving, give a mild extra push to escape the
    # plateau; if it is dropping fast, relax to avoid overshoot.
    stagnation_factor = 1.0
    if overflow_history is not None and len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-4:          # essentially stuck
            stagnation_factor = 1.04
        elif improvement > 0.05:        # improving quickly
            stagnation_factor = 0.99

    # --- Gradient-norm safeguard ---
    # Exploding gradients are the usual cause of divergence; damp growth when
    # the gradient norm is very large.
    grad_factor = 1.0
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            grad_factor = 0.97

    mult = base_mult * overflow_factor * stagnation_factor * grad_factor
    # Keep the per-step multiplier in a safe band.
    mult = min(max(mult, 0.95), 1.08)

    cur = current_lambda if current_lambda == current_lambda and current_lambda > 0.0 else 1.0
    new_lambda = cur * mult

    # --- Hard clamp to the required range ---
    return float(min(max(new_lambda, 0.01), 50.0))