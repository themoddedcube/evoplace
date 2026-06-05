def lambda_schedule(
    iteration: int,            # current optimization step
    overflow: float,           # current density overflow in [0, 1]
    overflow_history: list,    # overflow values at previous hook calls
    gradient_norm: float,      # L2 norm of the position gradient
    current_lambda: float,     # current density weight
) -> float:
    UPPER_PCOF = 1.05

    # sanitize overflow (guard NaN)
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # DREAMPlace default ramp: aggressive while still spreading
    base = max(0.9999 ** float(iteration), 0.98)
    mu = UPPER_PCOF * base

    # Once overflow falls under the legalization target, taper mu from the
    # full ramp down toward a gentle hold. mu is kept >= ~1.0 at all times,
    # so lambda never decreases -> placement cannot un-spread (overflow stays
    # legal and passes the fitness gate). Reducing the late-stage push lets the
    # accurate low-gamma gradient sharpen HPWL instead of over-spreading cells.
    TARGET = 0.10
    if of < TARGET:
        frac = of / TARGET                 # 1 at target -> 0 as it fully legalizes
        hold = 1.0 + 0.006 * base          # gentle positive bias to hold the spread
        mu = hold + (mu - hold) * frac

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))