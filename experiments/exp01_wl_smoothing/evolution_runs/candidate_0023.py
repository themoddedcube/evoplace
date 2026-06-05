import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # Progress in [0, 1]; guard against degenerate total_iterations.
    T = max(1, total_iterations)
    t = min(max(iteration, 0), T) / T

    # Clamp overflow to a sane range (overflow can drift slightly out of [0,1]).
    ov = min(1.0, max(0.0, overflow))

    # --- Base schedule: high gamma early -> low gamma late (cosine annealing) ---
    # Cosine gives a smooth, slow start and a gentle tail for fine-tuning.
    g_hi, g_lo = 8.0, 0.5
    cos = 0.5 * (1.0 + math.cos(math.pi * t))          # 1 at start -> 0 at end
    base = g_lo + (g_hi - g_lo) * cos                  # 8.0 -> 0.5

    # --- Overflow-adaptive term ---
    # While cells are still spread out (high overflow) we want smoother gradients,
    # so push gamma up; as the layout settles (low overflow) let it relax.
    # DREAMPlace-style log spacing keeps the response well-conditioned.
    adapt = 10.0 ** ((ov - 0.1) * (20.0 / 9.0) - 1.0)  # ~0.1 .. ~10 over ov in [0.1,1]

    # Blend: lean on overflow early (placement still moving), on the
    # iteration schedule late (annealing toward accurate HPWL).
    w = t                                              # 0 early -> 1 late
    gamma = (1.0 - w) * (0.5 * base + 0.5 * 4.0 * adapt) + w * base

    # --- Convergence damping from HPWL history ---
    # If wirelength has plateaued, drop gamma faster to sharpen the approximation.
    if hpwl_history and len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        prev = sum(recent[:2]) / 2.0
        last = sum(recent[2:]) / 2.0
        if prev > 0 and abs(last - prev) / prev < 1e-3:
            gamma *= 0.7

    # Never let it go fully smooth in the final stretch.
    if t > 0.85:
        gamma = min(gamma, 1.5)

    return min(50.0, max(0.01, gamma))