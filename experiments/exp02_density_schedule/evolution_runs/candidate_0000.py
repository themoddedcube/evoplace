"""
Seed lambda (density weight) schedule for Exp 2.

Reproduces DREAMPlace's default HPWL-based multiplicative update as closely
as the hook signature allows. The true default is
    mu = UPPER_PCOF * max(0.9999^iter, 0.98)            if HPWL improved
    mu = UPPER_PCOF * UPPER_PCOF^(-delta_hpwl/ref_hpwl)  otherwise (clamped)
    lambda *= mu
but delta_hpwl/ref_hpwl are not exposed to the hook, so this seed implements
the dominant (HPWL-improving) branch with UPPER_PCOF = 1.05. Parity with the
default trajectory is verified empirically by the seed sanity gate
(norm_hpwl must be 1.0 +/- 0.01 through the real cascade) before any
evolution run — do not skip it.
"""


def lambda_schedule(
    iteration: int,            # current optimization step
    overflow: float,           # current density overflow in [0, 1]
    overflow_history: list,    # overflow values at previous hook calls
    gradient_norm: float,      # L2 norm of the position gradient
    current_lambda: float,     # current density weight
) -> float:
    """Multiplicative density-weight ramp matching DREAMPlace's default
    improving-HPWL branch. Returns the NEW lambda (> 0)."""
    UPPER_PCOF = 1.05
    mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)
    return current_lambda * mu
