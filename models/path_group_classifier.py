"""
Path-group-aware net classification for timing-driven placement (Exp 4).

Classifies each net by the timing path group of its endpoints (FF->FF,
FF->Latch, Latch->FF, Latch->Latch half-cycle, etc.) and computes per-net
criticality weights inversely proportional to each group's slack budget.

These weights are written to placedb.net_weights before NonLinearPlace runs,
turning the standard WA-WL into a path-group-aware objective at no extra
GPU cost (Variant A integration).

Degrades gracefully to plain HPWL (all weights = 1.0) when no SDC or Liberty
is available, which is the case for ISPD 2015 benchmarks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path group taxonomy
# ---------------------------------------------------------------------------

class PathGroup(IntEnum):
    FF_FF         = 0  # DFF → DFF, budget T − t_setup − t_skew
    FF_LATCH      = 1  # DFF → latch, budget T/2 − t_setup_latch − t_skew
    LATCH_FF      = 2  # latch → DFF, budget T/2 − t_cq − t_skew
    LATCH_LATCH   = 3  # latch → latch; conservative: treated as half-cycle
    ASYNC         = 4  # false path or no clock endpoint; weight = 1.0 (skip)
    MULTICYCLE    = 5  # set_multicycle_path N; budget = N*T
    UNCONSTRAINED = 6  # no SDC constraint; weight = 1.0 (plain HPWL fallback)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PathGroupConfig:
    clock_period_ps: float = 4000.0   # overridden by SDC parse
    t_setup_ff_ps: float = 70.0       # FF D-pin setup (Liberty typical NanGate45)
    t_setup_latch_ps: float = 80.0    # latch D-pin setup
    t_cq_budget_ps: float = 250.0     # worst-case latch clock-to-Q
    t_skew_frac: float = 0.15         # fraction of T reserved for skew+insertion
    max_net_weight: float = 10.0
    base_net_weight: float = 1.0
    gamma_timing: float = 1.0         # WA-WL smoothness for timing term
    timing_loss_alpha: float = 0.5    # α multiplier on L_timing
    multicycle_n: int = 2             # default N when not parsed from SDC


@dataclass
class PathGroupData:
    """All precomputed timing data for one benchmark; passed to make_timing_hook."""
    net_groups: np.ndarray        # (M,) int8 — PathGroup per net
    net_weights: np.ndarray       # (M,) float32 — static criticality weights
    group_budgets_ps: Dict[int, float]
    config: PathGroupConfig
    has_timing_constraints: bool = False
    stats: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Liberty cell-type parser
# ---------------------------------------------------------------------------

def parse_liberty_cell_types(lib_path: str) -> Dict[str, str]:
    """
    Minimal Liberty parser. Returns {cell_master_name: "FF" | "LATCH" | "COMB"}.

    Only reads cell(...){...} blocks and checks for ff/latch group keywords.
    Handles nested braces and C-style comments.
    """
    cell_types: Dict[str, str] = {}
    try:
        with open(lib_path, "r", errors="replace") as fh:
            text = fh.read()
    except OSError:
        logger.warning(f"Cannot read Liberty file: {lib_path}")
        return cell_types

    # Strip C-style block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Strip line comments
    text = re.sub(r"//[^\n]*", "", text)

    i = 0
    n = len(text)
    while i < n:
        # Find "cell ("
        m = re.search(r'\bcell\s*\(\s*(\w+)\s*\)', text[i:])
        if not m:
            break
        cell_name = m.group(1)
        block_start = i + m.end()
        # Find matching opening brace
        brace_pos = text.find("{", block_start)
        if brace_pos == -1:
            break

        # Walk to matching close brace
        depth = 0
        j = brace_pos
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        cell_body = text[brace_pos:j + 1]
        # Determine sequential type from body
        if re.search(r'\blatch\s*\(', cell_body):
            cell_types[cell_name] = "LATCH"
        elif re.search(r'\bff\s*\(', cell_body):
            cell_types[cell_name] = "FF"
        else:
            cell_types[cell_name] = "COMB"

        i = j + 1

    logger.debug(f"Liberty: {len(cell_types)} cells parsed from {Path(lib_path).name}")
    return cell_types


# ---------------------------------------------------------------------------
# SDC parser (clock period only)
# ---------------------------------------------------------------------------

def parse_sdc_clock_period(sdc_path: Optional[str]) -> Optional[float]:
    """
    Extract clock period in picoseconds from SDC create_clock statement.
    SDC periods are in nanoseconds.
    """
    if sdc_path is None:
        return None
    try:
        with open(sdc_path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if "create_clock" in line and "-period" in line:
                    m = re.search(r"-period\s+([\d.]+)", line)
                    if m:
                        return float(m.group(1)) * 1000.0  # ns → ps
    except OSError:
        logger.warning(f"Cannot read SDC: {sdc_path}")
    return None


# ---------------------------------------------------------------------------
# Slack budget table
# ---------------------------------------------------------------------------

def build_group_budgets(cfg: PathGroupConfig) -> Dict[int, float]:
    """Compute required-time budget (ps) for each PathGroup."""
    T = cfg.clock_period_ps
    t_sk = cfg.t_skew_frac * T
    return {
        PathGroup.FF_FF:        T - cfg.t_setup_ff_ps - t_sk,
        PathGroup.FF_LATCH:     T / 2 - cfg.t_setup_latch_ps - t_sk,
        PathGroup.LATCH_FF:     T / 2 - cfg.t_cq_budget_ps - t_sk,
        PathGroup.LATCH_LATCH:  T / 2 - cfg.t_cq_budget_ps - cfg.t_setup_latch_ps - t_sk,
        PathGroup.ASYNC:        0.0,
        PathGroup.MULTICYCLE:   cfg.multicycle_n * T - cfg.t_setup_ff_ps - t_sk,
        PathGroup.UNCONSTRAINED: 0.0,
    }


# ---------------------------------------------------------------------------
# Net classification
# ---------------------------------------------------------------------------

def _get_cell_master(placedb, node_id: int) -> str:
    """Return the Liberty cell master name for a placement node."""
    # Try DREAMPlace rawdb C++ accessor first
    try:
        name = placedb.rawdb.cellTypeName(node_id)
        if isinstance(name, bytes):
            name = name.decode()
        return name
    except Exception:
        pass
    # Fallback: instance name often encodes master (e.g. "DFF_X1_i42" → "DFF_X1")
    raw = placedb.node_names[node_id]
    inst_name = raw.decode() if isinstance(raw, bytes) else str(raw)
    # Strip trailing _<digits> which is a common instance suffix
    return re.sub(r"_\d+$", "", inst_name.split("/")[-1])


def classify_nets(
    placedb,
    cell_type_map: Dict[str, str],
    sdc_path: Optional[str],
    cfg: Optional[PathGroupConfig] = None,
) -> PathGroupData:
    """
    Classify all nets and compute static criticality weights.

    Returns PathGroupData with net_groups and net_weights arrays, ready
    to be written to placedb.net_weights.
    """
    if cfg is None:
        cfg = PathGroupConfig()

    # Override clock period from SDC if available
    sdc_period = parse_sdc_clock_period(sdc_path)
    if sdc_period is not None:
        cfg = PathGroupConfig(**{**cfg.__dict__, "clock_period_ps": sdc_period})
        has_constraints = True
    else:
        has_constraints = False

    if not has_constraints or not cell_type_map:
        logger.info("Path-group classifier: no SDC/Liberty — all nets unconstrained (plain HPWL)")
        num_nets = len(placedb.net_names)
        return PathGroupData(
            net_groups=np.full(num_nets, PathGroup.UNCONSTRAINED, dtype=np.int8),
            net_weights=np.ones(num_nets, dtype=np.float32),
            group_budgets_ps=build_group_budgets(cfg),
            config=cfg,
            has_timing_constraints=False,
            stats={"num_nets": num_nets, "constrained_nets": 0},
        )

    # Build node_id → sequential type
    num_nodes = len(placedb.node_names)
    node_seq = np.zeros(num_nodes, dtype=np.int8)  # 0=COMB, 1=FF, 2=LATCH
    _TYPE_MAP = {"FF": 1, "LATCH": 2, "COMB": 0}
    for nid in range(num_nodes):
        master = _get_cell_master(placedb, nid)
        node_seq[nid] = _TYPE_MAP.get(cell_type_map.get(master, "COMB"), 0)

    # Classify each net
    num_nets = len(placedb.net_names)
    net_groups = np.full(num_nets, PathGroup.UNCONSTRAINED, dtype=np.int8)

    for net_id in range(num_nets):
        pin_ids = placedb.net2pin_map[net_id]
        if len(pin_ids) < 2:
            continue

        driver_seq = -1
        sink_seqs: List[int] = []
        for pid in pin_ids:
            node = placedb.pin2node_map[pid]
            direction = placedb.pin_direct[pid]  # 1 = OUTPUT (driver)
            if direction == 1:
                driver_seq = int(node_seq[node])
            else:
                sink_seqs.append(int(node_seq[node]))

        if driver_seq == -1 or not sink_seqs:
            continue

        # Worst-case sink: latch > FF > comb (tighter budget = higher priority)
        worst_sink = max(sink_seqs)  # 2=LATCH > 1=FF > 0=COMB

        if driver_seq == 1 and worst_sink == 1:
            net_groups[net_id] = PathGroup.FF_FF
        elif driver_seq == 1 and worst_sink == 2:
            net_groups[net_id] = PathGroup.FF_LATCH
        elif driver_seq == 2 and worst_sink == 1:
            net_groups[net_id] = PathGroup.LATCH_FF
        elif driver_seq == 2 and worst_sink == 2:
            net_groups[net_id] = PathGroup.LATCH_LATCH
        # COMB driver/sink stays UNCONSTRAINED (combinational cloud, no direct seq budget)

    budgets = build_group_budgets(cfg)
    net_weights = _compute_static_weights(net_groups, budgets, cfg)

    # Gather stats for logging
    group_counts = {g.name: int((net_groups == g).sum()) for g in PathGroup}
    constrained = int(np.sum(net_weights > 1.05))
    logger.info(
        f"Path-group classifier: T={cfg.clock_period_ps:.0f}ps, "
        f"constrained nets={constrained}/{num_nets}, "
        f"FF_FF={group_counts['FF_FF']}, "
        f"LATCH_LATCH={group_counts['LATCH_LATCH']}"
    )

    return PathGroupData(
        net_groups=net_groups,
        net_weights=net_weights,
        group_budgets_ps=budgets,
        config=cfg,
        has_timing_constraints=True,
        stats={**group_counts, "num_nets": num_nets, "constrained_nets": constrained},
    )


def _compute_static_weights(
    net_groups: np.ndarray,
    budgets: Dict[int, float],
    cfg: PathGroupConfig,
) -> np.ndarray:
    """Assign w(net) = T / RT(group), clamped to [1.0, max_weight]."""
    T = cfg.clock_period_ps
    weights = np.ones(len(net_groups), dtype=np.float32)
    for g_int, rt in budgets.items():
        if rt <= 0:
            continue
        mask = net_groups == g_int
        if not mask.any():
            continue
        ratio = float(T) / float(rt)
        weights[mask] = np.clip(cfg.base_net_weight * ratio, 1.0, cfg.max_net_weight)
    return weights
