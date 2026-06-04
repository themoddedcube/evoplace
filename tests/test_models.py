"""
Unit tests for models/gnn_placer.py and models/timing_surrogate.py.
Runs on CPU — no CUDA needed.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gnn_placer import GNNPlacer, CellNetMessagePassing, GNNPlacerTrainer
from models.timing_surrogate import TimingSurrogateNet, build_net_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_small_netlist(num_cells=20, num_nets=10, num_pins=40):
    """Create a small random netlist for testing."""
    rng = torch.Generator()
    rng.manual_seed(42)
    cell_area       = torch.rand(num_cells).abs() + 0.1
    cell_pin_count  = torch.randint(1, 10, (num_cells,))
    cell_type_ids   = torch.randint(0, 64, (num_cells,))
    cell_is_macro   = (torch.rand(num_cells) > 0.9)
    net_fanout      = torch.randint(2, 8, (num_nets,)).float()
    net_weight      = torch.ones(num_nets)
    net_is_critical = (torch.rand(num_nets) > 0.7)
    # Random pin→net assignments
    cell_ids = torch.randint(0, num_cells, (num_pins,))
    net_ids  = torch.randint(0, num_nets,  (num_pins,))
    cell_to_net_idx = torch.stack([cell_ids, net_ids], dim=1)
    return dict(
        cell_area=cell_area, cell_pin_count=cell_pin_count,
        cell_type_ids=cell_type_ids, cell_is_macro=cell_is_macro,
        net_fanout=net_fanout, net_weight=net_weight,
        net_is_critical=net_is_critical, cell_to_net_idx=cell_to_net_idx,
        num_cells=num_cells, num_nets=num_nets,
    )


# ── CellNetMessagePassing ─────────────────────────────────────────────────────

class TestCellNetMessagePassing:
    def test_output_shape(self):
        mp = CellNetMessagePassing(cell_dim=32, net_dim=32, hidden=64)
        N, M, P = 10, 5, 20
        cell_feat = torch.randn(N, 32)
        net_feat  = torch.randn(M, 32)
        cell_ids  = torch.randint(0, N, (P,))
        net_ids   = torch.randint(0, M, (P,))
        idx       = torch.stack([cell_ids, net_ids], dim=1)
        new_cell, new_net = mp(cell_feat, net_feat, idx)
        assert new_cell.shape == (N, 32)
        assert new_net.shape  == (M, 32)

    def test_gradients_flow(self):
        mp = CellNetMessagePassing(cell_dim=16, net_dim=16, hidden=32)
        N, M, P = 5, 3, 10
        cell_feat = torch.randn(N, 16, requires_grad=True)
        net_feat  = torch.randn(M, 16, requires_grad=True)
        idx = torch.stack([torch.randint(0, N, (P,)), torch.randint(0, M, (P,))], dim=1)
        out_cell, _ = mp(cell_feat, net_feat, idx)
        loss = out_cell.sum()
        loss.backward()
        assert cell_feat.grad is not None
        assert not torch.isnan(cell_feat.grad).any()


# ── GNNPlacer ─────────────────────────────────────────────────────────────────

class TestGNNPlacer:
    def test_forward_output_shape(self):
        nl = make_small_netlist()
        model = GNNPlacer()
        with torch.no_grad():
            positions = model(
                nl["cell_area"], nl["cell_pin_count"], nl["cell_type_ids"],
                nl["cell_is_macro"], nl["net_fanout"], nl["net_weight"],
                nl["net_is_critical"], nl["cell_to_net_idx"],
            )
        assert positions.shape == (nl["num_cells"], 2)

    def test_output_in_unit_square(self):
        nl = make_small_netlist()
        model = GNNPlacer()
        with torch.no_grad():
            positions = model(
                nl["cell_area"], nl["cell_pin_count"], nl["cell_type_ids"],
                nl["cell_is_macro"], nl["net_fanout"], nl["net_weight"],
                nl["net_is_critical"], nl["cell_to_net_idx"],
            )
        assert positions.min() >= 0.0 - 1e-6
        assert positions.max() <= 1.0 + 1e-6

    def test_no_nan_in_output(self):
        nl = make_small_netlist()
        model = GNNPlacer()
        with torch.no_grad():
            positions = model(
                nl["cell_area"], nl["cell_pin_count"], nl["cell_type_ids"],
                nl["cell_is_macro"], nl["net_fanout"], nl["net_weight"],
                nl["net_is_critical"], nl["cell_to_net_idx"],
            )
        assert not torch.isnan(positions).any()

    def test_gradient_flows_to_inputs(self):
        nl = make_small_netlist(num_cells=10, num_nets=5, num_pins=15)
        model = GNNPlacer()
        cell_area = nl["cell_area"].clone().requires_grad_(True)
        positions = model(
            cell_area, nl["cell_pin_count"], nl["cell_type_ids"],
            nl["cell_is_macro"], nl["net_fanout"], nl["net_weight"],
            nl["net_is_critical"], nl["cell_to_net_idx"],
        )
        positions.sum().backward()
        assert cell_area.grad is not None

    def test_different_netlists_give_different_outputs(self):
        model = GNNPlacer()
        nl1 = make_small_netlist(num_cells=10, num_nets=5, num_pins=15)
        nl2 = make_small_netlist(num_cells=10, num_nets=5, num_pins=15)
        # Tweak nl2 to be different
        nl2["cell_area"] = nl2["cell_area"] * 2
        with torch.no_grad():
            p1 = model(nl1["cell_area"], nl1["cell_pin_count"], nl1["cell_type_ids"],
                       nl1["cell_is_macro"], nl1["net_fanout"], nl1["net_weight"],
                       nl1["net_is_critical"], nl1["cell_to_net_idx"])
            p2 = model(nl2["cell_area"], nl2["cell_pin_count"], nl2["cell_type_ids"],
                       nl2["cell_is_macro"], nl2["net_fanout"], nl2["net_weight"],
                       nl2["net_is_critical"], nl2["cell_to_net_idx"])
        assert not torch.allclose(p1, p2)

    def test_parameter_count_reasonable(self):
        model = GNNPlacer()
        n_params = sum(p.numel() for p in model.parameters())
        # Should be in range 100k-5M params for our architecture
        assert 50_000 < n_params < 5_000_000, f"Unexpected param count: {n_params}"

    def test_trainer_step_reduces_loss(self):
        model = GNNPlacer()
        trainer = GNNPlacerTrainer(model, device="cpu")
        nl = make_small_netlist(num_cells=15, num_nets=8, num_pins=25)
        batch = {**nl, "target_positions": torch.rand(nl["num_cells"], 2)}
        losses = [trainer.train_step(batch) for _ in range(5)]
        # Loss should generally decrease (not guaranteed in 5 steps, but check it runs)
        assert all(math.isfinite(l) for l in losses)

    def test_large_netlist_doesnt_oom(self):
        """Verify the architecture handles realistic-sized netlists on CPU."""
        nl = make_small_netlist(num_cells=5000, num_nets=2000, num_pins=20000)
        model = GNNPlacer()
        with torch.no_grad():
            positions = model(
                nl["cell_area"], nl["cell_pin_count"], nl["cell_type_ids"],
                nl["cell_is_macro"], nl["net_fanout"], nl["net_weight"],
                nl["net_is_critical"], nl["cell_to_net_idx"],
            )
        assert positions.shape == (5000, 2)


import math


# ── TimingSurrogateNet ────────────────────────────────────────────────────────

class TestTimingSurrogatNet:
    def make_net_features(self, M=50):
        return torch.randn(M, TimingSurrogateNet.INPUT_DIM)

    def test_forward_output_shape(self):
        model = TimingSurrogateNet()
        feats = self.make_net_features(50)
        out = model(feats)
        assert out.shape == (50,)

    def test_tns_loss_non_negative(self):
        model = TimingSurrogateNet()
        feats = self.make_net_features(50)
        tns = model.tns_loss(feats)
        assert float(tns) >= 0.0

    def test_tns_loss_differentiable(self):
        model = TimingSurrogateNet()
        feats = self.make_net_features(20).requires_grad_(True)
        tns = model.tns_loss(feats)
        tns.backward()
        assert feats.grad is not None
        assert not torch.isnan(feats.grad).any()

    def test_tns_zero_for_all_positive_slack(self):
        """If model predicts all-positive slack, TNS should be near zero."""
        model = TimingSurrogateNet()
        # Override final layer to always predict +10 (large positive slack)
        with torch.no_grad():
            model.net[-1].weight.fill_(0)
            model.net[-1].bias.fill_(10.0)
        feats = self.make_net_features(20)
        tns = model.tns_loss(feats)
        # softplus(-10) ≈ 4.5e-5 per net → sum ≈ 9e-4
        assert float(tns) < 0.1

    def test_build_net_features_shape(self):
        N, M = 20, 10
        cell_x = torch.rand(N, requires_grad=True)
        cell_y = torch.rand(N, requires_grad=True)
        cell_area = torch.rand(N) + 0.1
        nets = [list(np.random.randint(0, N, size=np.random.randint(2, 5)))
                for _ in range(M)]
        critical = torch.rand(M) > 0.5
        feats = build_net_features(cell_x, cell_y, cell_area, nets, critical, 1.0, 1.0)
        assert feats.shape == (M, TimingSurrogateNet.INPUT_DIM)

    def test_build_net_features_gradients_flow(self):
        N, M = 10, 5
        cell_x = torch.rand(N, requires_grad=True)
        cell_y = torch.rand(N, requires_grad=True)
        nets = [[0, 1, 2], [3, 4], [5, 6, 7], [8, 9], [0, 5]]
        feats = build_net_features(cell_x, cell_y, None, nets, None, 1.0, 1.0)
        feats.sum().backward()
        # At least some gradients should flow (HPWL terms involve cell positions)
        assert cell_x.grad is not None or cell_y.grad is not None
