import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # progress in [0, 1]
    T = max(1, total_iterations)
    t = min(max(iteration, 0), T) / T

    # Base annealing: high gamma early (smooth gradients, let cells cluster),
    # low gamma late (accurate HPWL, fine-tuning). Cosine annealing on a log
    # scale between gamma_hi and gamma_lo gives a gentle early hold and a soft
    # landing rather than an abrupt drop.
    g_hi = 8.0
    g_lo = 0.5
    cos = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
    log_gamma = math.log(g_lo) + (math.log(g_hi) - math.log(g_lo)) * cos
    gamma = math.exp(log_gamma)

    # Overflow-adaptive coupling: placement quality is governed far more by
    # overflow than by raw iteration count. While cells are still badly spread
    # (high overflow) we keep gamma elevated to preserve smooth, far-reaching
    # gradients; as the layout legalizes (overflow -> ~0.1) we let gamma fall so
    # the WA-WL surrogate sharpens toward the true HPWL.
    ov = min(max(overflow, 0.0), 1.0)
    # map overflow (target floor ~0.1) into a multiplicative factor in ~[0.6, 1.8]
    ov_factor = 0.6 + 1.2 * min(1.0, max(0.0, (ov - 0.1) / 0.9))
    gamma *= ov_factor

    # Plateau detection: if recent HPWL has stopped improving, nudge gamma down
    # to chase a more accurate objective and escape the surrogate's bias.
    if hpwl_history is not None and len(hpwl_history) >= 6:
        recent = hpwl_history[-3:]
        prev = hpwl_history[-6:-3]
        avg_recent = sum(recent) / len(recent)
        avg_prev = sum(prev) / len(prev)
        if avg_prev > 0.0:
            rel_improve = (avg_prev - avg_recent) / abs(avg_prev)
            if rel_improve < 1e-4:  # stalled
                gamma *= 0.8

    # Late-stage sharpening floor: in the final phase, force gamma low enough
    # that the surrogate tracks true HPWL for the final convergence.
    if t > 0.85:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))