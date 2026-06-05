import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    # --- defensive normalization -------------------------------------------
    t = iteration / max(1, total_iterations)
    t = min(1.0, max(0.0, t))

    ov = overflow
    if ov != ov:            # NaN guard -> assume fully overlapping
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    # --- overflow-adaptive base (RePlAce/DREAMPlace style) -----------------
    # high overflow (cells clustered) -> large smoothing gamma
    # low  overflow (spread out)      -> small accurate gamma
    # maps ov in [0.1, 1.0] log-linearly to ~[0.8, 50]
    base = 8.0 * 10.0 ** (1.8 * (ov - 0.1))

    # --- cosine-annealed iteration envelope --------------------------------
    # smooth high->low fallback that does not depend on the overflow signal
    hi, lo = 8.0, 0.5
    cos_env = lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * t))

    # --- blend: trust overflow early, anneal late --------------------------
    gamma = (1.0 - t) * base + t * cos_env

    # --- stagnation detection: sharpen for final HPWL accuracy -------------
    if len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-6:] if h == h and abs(h) != float("inf")]
        if len(recent) >= 4:
            m = max(recent)
            if m > 0.0 and (m - min(recent)) / m < 1e-3:
                gamma *= 0.6      # plateau -> reduce gamma toward true HPWL

    # --- hard floor in the last 10% to lock in fine placement --------------
    if t > 0.9:
        gamma = min(gamma, 1.5)

    if gamma != gamma or abs(gamma) == float("inf"):
        gamma = 1.0

    return min(50.0, max(0.01, gamma))