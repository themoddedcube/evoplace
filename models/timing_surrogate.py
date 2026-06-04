"""
Differentiable TNS surrogate model (Experiment 4).

A lightweight MLP that approximates post-route TNS from placement features.
This enables gradient-based TNS optimization during global placement.

Input features (per net):
- Net HPWL (current placement)
- Net fanout
- Driver cell area, sink cell areas (mean, max)
- Is-on-critical-path flag
- Net distance to die boundary

Output: per-net timing slack approximation (negative = violation)
Total TNS = sum of clipped negative slacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class TimingSurrogateNet(nn.Module):
    """
    MLP surrogate for per-net timing slack prediction.

    Trained on (placement features, actual TNS) pairs from the ICCAD 2015 benchmarks.
    Used as a differentiable TNS proxy during global placement.
    """

    INPUT_DIM = 8  # see features below
    HIDDEN_DIM = 256

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # per-net slack
        )

    def forward(self, net_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            net_features: (M, 8) tensor of per-net features:
                [0] hpwl (normalized)
                [1] fanout (log-normalized)
                [2] driver_area (log-normalized)
                [3] mean_sink_area (log-normalized)
                [4] max_sink_area (log-normalized)
                [5] is_critical_path (0/1)
                [6] dist_to_boundary_x (normalized)
                [7] dist_to_boundary_y (normalized)

        Returns:
            (M,) predicted slack per net (negative = timing violation)
        """
        return self.net(net_features).squeeze(-1)

    def tns_loss(self, net_features: torch.Tensor) -> torch.Tensor:
        """
        Differentiable TNS = sum of negative predicted slacks.
        Suitable as a loss term in placement optimization.
        """
        slacks = self.forward(net_features)
        # Soft-plus approximation of max(0, -slack) for differentiability
        violations = F.softplus(-slacks)
        return violations.sum()


def build_net_features(
    cell_x: torch.Tensor,
    cell_y: torch.Tensor,
    cell_area: torch.Tensor,
    net_cell_indices,
    critical_net_flags: Optional[torch.Tensor],
    die_w: float,
    die_h: float,
) -> torch.Tensor:
    """
    Compute per-net features from current placement.

    This function must be differentiable w.r.t. cell_x, cell_y
    so gradients flow back to cell positions.
    """
    num_nets = len(net_cell_indices)
    features = torch.zeros(num_nets, TimingSurrogateNet.INPUT_DIM, device=cell_x.device)

    for i, cells in enumerate(net_cell_indices):
        if len(cells) < 2:
            continue
        cells_t = torch.tensor(cells, dtype=torch.long, device=cell_x.device)
        xs = cell_x[cells_t]
        ys = cell_y[cells_t]

        # HPWL (differentiable)
        hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
        features[i, 0] = hpwl / (die_w + die_h)

        # Fanout
        features[i, 1] = torch.log(torch.tensor(float(len(cells))) + 1)

        # Driver / sink areas (first cell = driver by convention)
        driver_area = cell_area[cells_t[0]] if cell_area is not None else torch.tensor(1.0)
        sink_areas = cell_area[cells_t[1:]] if cell_area is not None else torch.ones(len(cells) - 1)
        features[i, 2] = torch.log(driver_area + 1)
        features[i, 3] = torch.log(sink_areas.mean() + 1)
        features[i, 4] = torch.log(sink_areas.max() + 1)

        # Critical path flag
        if critical_net_flags is not None:
            features[i, 5] = critical_net_flags[i].float()

        # Distance to boundary (of centroid)
        cx = xs.mean() / die_w
        cy = ys.mean() / die_h
        features[i, 6] = torch.min(cx, 1 - cx)
        features[i, 7] = torch.min(cy, 1 - cy)

    return features


def load_timing_surrogate(checkpoint_path: str, device: str = "cpu") -> TimingSurrogateNet:
    """Load a trained timing surrogate from checkpoint."""
    model = TimingSurrogateNet()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model
