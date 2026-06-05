import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """ ... """

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Overflow is the primary convergence signal: high overflow means cells are
    # still spreading/clustering (favor smooth gradients), low overflow means the
    # placement is nearly legal (favor an accurate HPWL approximation). Interpolate
    # geometrically in log-space between the low and high gamma endpoints.
    ov_term = gamma_low * (gamma_high / gamma_low) ** ov

    # Iteration-based exponential decay so gamma still anneals when overflow
    # plateaus, guaranteeing fine-tuning at the end of the run.
    iter_term = gamma_high * (gamma_low / gamma_high) ** progress

    # Early iterations trust the time schedule; later iterations let the measured
    # overflow drive sharpening. Smooth blend weighted by progress.
    w = progress * progress * (3.0 - 2.0 * progress)  # smoothstep
    gamma = (1.0 - w) * iter_term + w * ov_term

    # HPWL-trajectory adaptation.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-5:] if h == h and abs(h) != float("inf")]
        if len(recent) >= 2:
            prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
            best_recent = min(recent)
            # Stagnating: improvement < 0.1% -> sharpen to escape the plateau.
            if prev == prev and abs(prev) != float("inf") and prev > 0:
                if (prev - best_recent) / prev < 1e-3:
                    gamma *= 0.8
            # Diverging: wirelength climbing -> smooth gradients to re-stabilize.
            if recent[-1] > recent[0] * 1.02:
                gamma *= 1.3

    # Hard low-gamma regime at the very end for accurate final HPWL.
    if progress > 0.9:
        gamma = min(gamma, 0.8)

    # Guard against any non-finite result before clamping.
    if not (gamma == gamma) or abs(gamma) == float("inf"):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))