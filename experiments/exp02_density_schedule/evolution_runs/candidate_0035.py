def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Density-penalty (lambda) schedule for DREAMPlace-style placement.
    # Strategy: multiplicative growth of the density penalty, modulated by
    # how fast overflow is improving. Push hard while cells are still
    # overlapping (high overflow), ease off once spreading stalls or
    # overflow is already low so wirelength can be fine-tuned.

    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Clamp inputs defensively (NaN/inf -> safe values) so we never return inf.
    if not (current_lambda == current_lambda) or current_lambda in (
        float("inf"),
        float("-inf"),
    ):
        current_lambda = 1.0
    if not (overflow == overflow):
        overflow = 1.0
    overflow = min(max(overflow, 0.0), 1.0)

    # Base decaying multiplier (as in classic ePlace/DREAMPlace).
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive growth rate: interpolate between LOWER and UPPER
    # based on current overflow. High overflow -> aggressive penalty growth.
    growth = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * overflow

    # Trend term: if overflow is dropping steadily, growth is working, keep
    # momentum; if it has plateaued, accelerate to break the stall.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[0] - recent[-1]  # positive = improving
        if delta < 1e-4 and overflow > 0.1:
            growth *= 1.02  # plateau: push harder
        elif delta > 0.02:
            growth *= 0.995  # improving fast: relax slightly

    mu = growth * base

    new_lambda = current_lambda * mu

    # Hard bound required by the contract.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0

    return float(new_lambda)