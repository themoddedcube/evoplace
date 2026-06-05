import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma with iteration annealing and plateau-aware fine-tuning."""

    # Guard against degenerate inputs.
    T = max(1, int(total_iterations))
    t = min(max(0, int(iteration)), T)
    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))
    progress = t / T

    # --- Primary driver: overflow ---------------------------------------
    # When cells are still spread out (high overflow) we want large gamma for
    # smooth, well-conditioned gradients that let cells migrate globally.
    # As the layout legalizes (overflow -> 0) we drop gamma toward an accurate
    # HPWL approximation. Map overflow in [0.1, 1.0] -> gamma in ~[0.5, 8.0]
    # geometrically (linear in log-space), which matches DREAMPlace practice.
    lo_ov, hi_ov = 0.10, 1.00
    log_lo, log_hi = math.log10(0.5), math.log10(8.0)
    frac = (ov - lo_ov) / (hi_ov - lo_ov)
    frac = min(1.0, max(0.0, frac))
    log_gamma = log_lo + frac * (log_hi - log_lo)
    gamma = 10.0 ** log_gamma

    # --- Iteration annealing --------------------------------------------
    # Even if overflow lingers, force a gentle cosine decay over the run so the
    # late phase always sharpens the objective. Caps gamma by a schedule that
    # starts near 8.0 and anneals to ~0.5.
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    iter_cap = 0.5 + (8.0 - 0.5) * cos
    gamma = min(gamma, iter_cap)

    # --- Plateau-aware fine-tuning --------------------------------------
    # If HPWL has stopped improving late in the run, push gamma lower to extract
    # the most accurate wirelength at the cost of noisier gradients.
    if progress > 0.6 and hpwl_history and len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        prev = hpwl_history[-5] if len(hpwl_history) >= 5 else recent[0]
        best_recent = min(recent)
        denom = abs(prev) + 1e-12
        rel_impr = (prev - best_recent) / denom
        if rel_impr < 1e-3:
            gamma *= 0.7

    # --- Late-stage floor ------------------------------------------------
    # In the final 10% lock in an accurate approximation.
    if progress > 0.9:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))