"""
GNN-based warm initialization for global placement (Experiment 3).

Architecture: Heterogeneous GNN on cell-net hypergraph.
- Cell nodes: features = [area, pin_count, cell_type_embedding(16)]
- Net nodes: features = [fanout, net_weight, is_critical_path]
- Cell→Net and Net→Cell message passing
- 4 layers, hidden_dim=128
- Output: (x_pred, y_pred) per cell, normalized to [0, 1]

Training: Supervised on DREAMPlace final placements.
Loss: MSE + boundary penalty (keep predictions in [0,1]).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CellNetMessagePassing(nn.Module):
    """One round of cell→net→cell message passing."""

    def __init__(self, cell_dim: int, net_dim: int, hidden: int):
        super().__init__()
        # Cell→Net aggregation
        self.cell_to_net = nn.Sequential(
            nn.Linear(cell_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, net_dim),
        )
        # Net→Cell aggregation
        self.net_to_cell = nn.Sequential(
            nn.Linear(net_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, cell_dim),
        )
        # Update MLPs
        self.cell_update = nn.Sequential(
            nn.Linear(cell_dim + cell_dim, cell_dim),
            nn.LayerNorm(cell_dim),
            nn.ReLU(),
        )
        self.net_update = nn.Sequential(
            nn.Linear(net_dim + net_dim, net_dim),
            nn.LayerNorm(net_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        cell_feat: torch.Tensor,      # (num_cells, cell_dim)
        net_feat: torch.Tensor,        # (num_nets, net_dim)
        cell_to_net_idx: torch.Tensor, # (num_pins, 2) — [cell_id, net_id]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nets = net_feat.shape[0]
        num_cells = cell_feat.shape[0]

        # Aggregate cell features into nets (mean pooling over pins)
        cell_msgs = self.cell_to_net(cell_feat[cell_to_net_idx[:, 0]])  # (num_pins, net_dim)
        net_agg = torch.zeros(num_nets, net_feat.shape[1], device=net_feat.device)
        net_agg.scatter_add_(0, cell_to_net_idx[:, 1].unsqueeze(1).expand_as(cell_msgs), cell_msgs)
        # Normalize by fanout
        counts = torch.bincount(cell_to_net_idx[:, 1], minlength=num_nets).float().clamp(min=1)
        net_agg = net_agg / counts.unsqueeze(1)

        # Aggregate net features into cells
        net_msgs = self.net_to_cell(net_feat[cell_to_net_idx[:, 1]])  # (num_pins, cell_dim)
        cell_agg = torch.zeros(num_cells, cell_feat.shape[1], device=cell_feat.device)
        cell_agg.scatter_add_(0, cell_to_net_idx[:, 0].unsqueeze(1).expand_as(net_msgs), net_msgs)
        pin_counts = torch.bincount(cell_to_net_idx[:, 0], minlength=num_cells).float().clamp(min=1)
        cell_agg = cell_agg / pin_counts.unsqueeze(1)

        # Update
        new_cell = self.cell_update(torch.cat([cell_feat, cell_agg], dim=-1))
        new_net = self.net_update(torch.cat([net_feat, net_agg], dim=-1))

        return new_cell, new_net


class GNNPlacer(nn.Module):
    """
    GNN that predicts (x, y) starting positions for global placement.

    Given a netlist graph, outputs a position for each cell that serves as
    warm initialization for DREAMPlace instead of uniform center initialization.
    """

    CELL_INPUT_DIM = 19   # area(1) + pin_count(1) + cell_type_emb(16) + is_macro(1)
    NET_INPUT_DIM = 3     # fanout(1) + net_weight(1) + is_critical(1)
    HIDDEN_DIM = 128
    NUM_LAYERS = 4

    def __init__(
        self,
        cell_input_dim: int = CELL_INPUT_DIM,
        net_input_dim: int = NET_INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_cell_types: int = 64,
    ):
        super().__init__()

        # Cell type embedding
        self.cell_type_emb = nn.Embedding(num_cell_types, 16)

        # Input projections
        self.cell_proj = nn.Linear(cell_input_dim, hidden_dim)
        self.net_proj = nn.Linear(net_input_dim, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            CellNetMessagePassing(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Output head: predict (x, y) in [0, 1]
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),  # outputs in [0, 1]
        )

    def forward(
        self,
        cell_area: torch.Tensor,        # (N,) cell areas
        cell_pin_count: torch.Tensor,   # (N,) pin counts
        cell_type_ids: torch.Tensor,    # (N,) integer type IDs
        cell_is_macro: torch.Tensor,    # (N,) bool
        net_fanout: torch.Tensor,       # (M,) net fanouts
        net_weight: torch.Tensor,       # (M,) net weights
        net_is_critical: torch.Tensor,  # (M,) bool
        cell_to_net_idx: torch.Tensor,  # (P, 2) [cell_id, net_id] per pin
    ) -> torch.Tensor:
        """Returns (N, 2) predicted positions, normalized to [0, 1]."""

        # Build cell features
        type_emb = self.cell_type_emb(cell_type_ids)  # (N, 16)
        cell_feat = torch.cat([
            cell_area.unsqueeze(1),
            cell_pin_count.unsqueeze(1).float(),
            type_emb,
            cell_is_macro.unsqueeze(1).float(),
        ], dim=-1)  # (N, 19)

        net_feat = torch.stack([
            net_fanout.float(),
            net_weight,
            net_is_critical.float(),
        ], dim=-1)  # (M, 3)

        # Project to hidden dim
        cell_h = F.relu(self.cell_proj(cell_feat))
        net_h = F.relu(self.net_proj(net_feat))

        # Message passing
        for layer in self.mp_layers:
            cell_h, net_h = layer(cell_h, net_h, cell_to_net_idx)

        # Predict positions
        positions = self.output_head(cell_h)  # (N, 2), values in [0, 1]
        return positions


class GNNPlacerTrainer:
    """Training loop for the GNN placement predictor."""

    def __init__(
        self,
        model: GNNPlacer,
        device: str = "cuda",
        learning_rate: float = 1e-3,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def loss_fn(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        cell_is_macro: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MSE loss + higher weight for macros (they're harder to place and more impactful).
        """
        mse = F.mse_loss(pred_positions, target_positions, reduction="none")  # (N, 2)
        weights = torch.ones(mse.shape[0], device=mse.device)
        if cell_is_macro is not None:
            weights[cell_is_macro] = 5.0  # 5× weight on macro positions
        loss = (mse.sum(dim=-1) * weights).mean()
        return loss

    def train_step(self, batch) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(
            batch["cell_area"].to(self.device),
            batch["cell_pin_count"].to(self.device),
            batch["cell_type_ids"].to(self.device),
            batch["cell_is_macro"].to(self.device),
            batch["net_fanout"].to(self.device),
            batch["net_weight"].to(self.device),
            batch["net_is_critical"].to(self.device),
            batch["cell_to_net_idx"].to(self.device),
        )
        target = batch["target_positions"].to(self.device)
        loss = self.loss_fn(pred, target, batch.get("cell_is_macro"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss)


def load_gnn_predictor(checkpoint_path: str, device: str = "cpu") -> GNNPlacer:
    """Load a trained GNN model from checkpoint."""
    model = GNNPlacer()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model
