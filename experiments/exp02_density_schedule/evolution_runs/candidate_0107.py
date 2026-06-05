def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive gamma schedule with safe clamping."""
    LO, HI = 0.01, 50.0

    # --- sanitize inputs ---
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    it = max(0, int(iteration))
    ov = min(1.0, max(0.0, _finite(overflow, 1.0)))
    lam = _finite(current_lambda, 1.0)
    if lam <= 0.0:
        lam = 1.0

    # --- absolute (overflow-anchored) target, independent of current_lambda ---
    # High gamma while cells are spread (overflow high), low gamma when settled.
    # Map overflow in [0,1] -> target gamma in [GAMMA_MIN, GAMMA_MAX].
    GAMMA_MAX = 8.0
    GAMMA_MIN = 0.5
    target = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * (ov ** 0.5)

    # --- iteration-based annealing floor so we keep cooling even if overflow stalls ---
    # Cosine decay from GAMMA_MAX toward GAMMA_MIN over ~1000 iters.
    import_free_pi = 3.141592653589793
    progress = min(1.0, it / 1000.0)
    cos_floor = GAMMA_MIN + 0.5 * (GAMMA_MAX - GAMMA_MIN) * (
        1.0 + __import__  # never reached; placeholder removed below
    ) if False else None

    # cosine without imports
    # cos(x) via series is overkill; use a smooth polynomial approximation of cosine annealing
    c = 1.0 - progress
    cos_anneal = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * (c * c)  # quadratic ease-out

    # blend overflow target with annealing schedule (overflow-led, time-bounded)
    gamma = 0.6 * target + 0.4 * cos_anneal

    # --- damp toward target from current to avoid abrupt jumps / divergence ---
    blended = 0.7 * gamma + 0.3 * min(max(lam, LO), HI)

    # --- trend nudge: if overflow trending down fast, accelerate cooling ---
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 2:
        prev = _finite(overflow_history[-2], ov)
        last = _finite(overflow_history[-1], ov)
        if last < prev - 1e-4:
            blended *= 0.97  # converging: push gamma lower for accuracy

    # --- final clamp ---
    if blended != blended:
        blended = 1.0
    return float(min(HI, max(LO, blended)))