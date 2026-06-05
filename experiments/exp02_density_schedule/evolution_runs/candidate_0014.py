def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # RePlAce/ePlace-style overflow-adaptive multiplicative penalty update.
    # lambda grows while cells remain spread-out (high overflow), then the
    # growth automatically tapers as overflow falls, letting the wirelength
    # term dominate for fine-tuning. All branches stay numerically bounded.

    UPPER_PCOF = 1.05      # max per-step growth
    LOWER_PCOF = 0.95      # min per-step growth (mild decay when nearly legal)
    OVF_TARGET = 0.10      # target overflow for convergence

    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Overflow trend: negative means we are improving (overflow dropping).
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        prev = min(max(prev if prev == prev else of, 0.0), 1.0)
        delta = of - prev
    else:
        delta = 0.0

    # Base multiplier from how far overflow is above target.
    # Far above target -> push lambda up; near/below target -> ease off.
    gap = (of - OVF_TARGET) / max(1.0 - OVF_TARGET, 1e-6)
    mu = 1.0 + 0.05 * gap                      # ~1.05 when full, ~0.995 when legal

    # If overflow is rising (placement spreading out), accelerate penalty.
    if delta > 0.0:
        mu *= (1.0 + min(delta, 0.2))

    # Gentle annealing so we always converge in late iterations.
    anneal = max(0.999 ** float(iteration), 0.97)
    mu *= anneal

    # Clamp the multiplier to a stable band.
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:               # NaN guard
        new_lambda = current_lambda

    return float(min(max(new_lambda, 0.01), 50.0))