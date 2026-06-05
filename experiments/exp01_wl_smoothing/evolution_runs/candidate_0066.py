import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine gamma schedule with plateau/divergence control."""

    # --- sanitize inputs (defend against None / NaN / inf) ---
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = _finite(iteration, 0.0) / total
    progress = min(1.0, max(0.0, progress))

    ov = _finite(overflow, 1.0)
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base decay: cosine annealing from high -> low ---
    # smooth, slow at the ends, fast in the middle; keeps gradients clean early
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    base = gamma_low + (gamma_high - gamma_low) * cos_factor

    # --- overflow coupling ---
    # when many bins are over-dense, cells are still tangled => keep gamma high.
    # when overflow is low, trust the geometry and sharpen the approximation.
    overflow_factor = 0.5 + 1.5 * (ov ** 1.25)
    gamma = base * overflow_factor

    # --- history-based adaptation ---
    hist = [h for h in (hpwl_history or []) if _finite(h, float("nan")) == _finite(h, float("nan"))]
    if len(hist) >= 5:
        recent = hist[-5:]
        prev = hist[-6] if len(hist) >= 6 else recent[0]
        best_recent = min(recent)

        # plateau: HPWL barely improving -> sharpen to chase accuracy
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.7

        # divergence: HPWL climbing -> smooth gradients to recover stability
        if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
            gamma *= 1.4

    # --- late-stage cap: force accurate HPWL for final fine-tuning ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 2.0)

    if gamma != gamma:  # final NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))