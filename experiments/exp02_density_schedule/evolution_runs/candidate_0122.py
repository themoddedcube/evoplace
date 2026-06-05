def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Overflow-adaptive multiplicative update of the density weight.
    # Push hard while bins are over-dense (cells still overlapping),
    # then ease the growth as the layout legalizes. Always clamp the
    # result so the weight cannot diverge to inf as it did before.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Early iterations should drive harder than late ones.
    base = max(0.9999 ** float(iteration), 0.98)

    # Normalize overflow to [0, 1].
    of = overflow if overflow == overflow else 1.0  # NaN guard
    of = min(max(of, 0.0), 1.0)

    # Overflow trend: accelerate if it is rising or stuck.
    if len(overflow_history) >= 2:
        delta = overflow_history[-1] - overflow_history[-2]
    else:
        delta = 0.0

    # Multiplier scales with remaining overflow: high overflow -> near
    # UPPER_PCOF, near-legal -> near LOWER_PCOF (gentle fine-tuning).
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of * base
    if delta > 0.0:           # overflow worsening -> spend more weight
        mu *= 1.01
    elif delta < -0.02:       # converging fast -> back off slightly
        mu = max(mu * 0.99, LOWER_PCOF)

    new_lambda = current_lambda * mu

    # Reject NaN/inf, then clamp to the legal range.
    if not (new_lambda == new_lambda) or new_lambda == float("inf"):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))