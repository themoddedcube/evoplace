def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight (lambda) update.

    Mirrors DREAMPlace's mu update: the multiplier shrinks toward 1.0 when
    overflow is falling quickly (let WL optimize) and grows toward UPPER_PCOF
    when overflow stalls or regresses (push density harder). Once the layout is
    nearly legal we stop inflating lambda so the solver can fine-tune wirelength.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Recent overflow reduction rate from history (positive == improving).
    if len(overflow_history) >= 2:
        delta_overflow = overflow_history[-2] - overflow_history[-1]
    else:
        delta_overflow = 0.0

    # Map reduction rate into [LOWER_PCOF, UPPER_PCOF]:
    #   fast drop  -> gentle push (mu ~ LOWER_PCOF)
    #   stall/regress -> aggressive push (mu ~ UPPER_PCOF)
    t = delta_overflow / 0.02
    if t > 1.0:
        t = 1.0
    elif t < 0.0:
        t = 0.0
    mu = UPPER_PCOF - (UPPER_PCOF - LOWER_PCOF) * t

    # Early iterations: keep gradients smooth by letting lambda climb a bit more;
    # this keeps cells clustered before fine-tuning.
    if iteration < 50 and overflow > 0.5:
        mu = max(mu, 1.02)

    # Nearly legal: cap growth so effective approximation stays accurate for WL.
    if overflow < 0.1:
        mu = min(mu, 1.0 + 0.5 * overflow)

    new_lambda = current_lambda * mu

    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return new_lambda