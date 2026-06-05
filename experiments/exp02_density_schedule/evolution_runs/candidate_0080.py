def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.005

    # Geometric base that anneals from aggressive -> gentle growth over time,
    # so density weight ramps hard early (cluster cells) and eases late (fine-tune HPWL).
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive coupling: while many bins are over-dense, push the penalty
    # harder; as the layout becomes legal, slow growth to recover accurate wirelength.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = max(0.0, min(1.0, of))
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Stall / progress detection from the overflow trajectory.
    if overflow_history and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-4:        # overflow plateaued -> accelerate penalty to break the stall
            pcof *= 1.03
        elif improvement > 2e-2:      # converging quickly -> already on track, don't overshoot
            pcof *= 0.98

    # Late-stage refinement: once overflow is low, bias toward a small, stable lambda
    # so gradients stay accurate for wirelength minimization.
    if of < 0.1:
        pcof = 1.0 + (pcof - 1.0) * 0.5

    # Gradient safeguard: damp growth when gradients blow up to keep the solve stable.
    if gradient_norm == gradient_norm and gradient_norm > 5.0:
        pcof = 1.0 + (pcof - 1.0) * 0.5

    mu = pcof * base
    new_lambda = current_lambda * mu

    return float(min(50.0, max(0.01, new_lambda)))