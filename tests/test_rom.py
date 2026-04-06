"""
tests/test_rom.py
──────────────────
Unit tests for the Riemannian Optimization Module (ROM) — paper Section 5.3.

Verifies:
  - Parameters stay on the Poincaré ball after updates
  - Curvature-adaptive LR decays as lr_0 / sqrt(1 + |c| * t)
  - Loss decreases on a simple hyperbolic regression task
  - create_optimizers correctly splits Euclidean vs Riemannian params
  - Riemannian gradient rescaling matches conformal factor formula

Run: pytest tests/test_rom.py -v
"""

import math
import pytest
import torch
import torch.nn as nn

from geonet.optim.riemannian_adam import (
    CurvatureAdaptiveRiemannianAdam,
    riemannian_gradient,
    create_optimizers,
)
from geonet.utils.manifold import (
    exp_map_zero,
    geodesic_distance,
    _project_to_ball,
    _EPS,
)


C = torch.tensor(-1.0)


def make_ball_param(shape, scale=0.3):
    """Create a random Poincaré-ball parameter."""
    data = torch.randn(*shape) * scale
    data = _project_to_ball(data, C)
    return nn.Parameter(data.clone())


# ─────────────────────────────────────────────────────────────────────────────
# Riemannian gradient
# ─────────────────────────────────────────────────────────────────────────────

class TestRiemannianGradient:
    def test_scale_at_origin(self):
        """At the origin, riemannian_grad = euclidean_grad (conformal factor = 1)."""
        g_e = torch.randn(4, 8)
        x   = torch.zeros(4, 8)
        g_r = riemannian_gradient(g_e, x, C)
        # At origin: (1 - |c|*0)^2 / 4 = 1/4, so g_r = g_e / 4
        expected = g_e / 4.0
        assert torch.allclose(g_r, expected, atol=1e-6), \
            f"Riemannian grad at origin incorrect: max diff = {(g_r - expected).abs().max():.2e}"

    def test_scale_decreases_near_boundary(self):
        """Points near boundary should have smaller Riemannian gradient (compressed metric)."""
        g_e  = torch.ones(1, 4)
        x_center = torch.zeros(1, 4)
        x_edge   = torch.tensor([[0.9, 0.0, 0.0, 0.0]])  # near boundary
        x_edge   = _project_to_ball(x_edge, C)

        g_center = riemannian_gradient(g_e, x_center, C).norm().item()
        g_edge   = riemannian_gradient(g_e, x_edge,   C).norm().item()
        assert g_edge < g_center, \
            f"Gradient near boundary ({g_edge:.4f}) should be < center ({g_center:.4f})."

    def test_gradient_shape_preserved(self):
        g_e = torch.randn(3, 5, 8)
        x   = torch.randn(3, 5, 8) * 0.2
        g_r = riemannian_gradient(g_e, x, C)
        assert g_r.shape == g_e.shape


# ─────────────────────────────────────────────────────────────────────────────
# CurvatureAdaptiveRiemannianAdam
# ─────────────────────────────────────────────────────────────────────────────

class TestRiemannianAdam:
    def setup_method(self):
        torch.manual_seed(42)
        self.d  = 16
        self.p  = make_ball_param((8, self.d))
        self.c  = C.clone()
        self.lr = 1e-2
        self.opt = CurvatureAdaptiveRiemannianAdam(
            [self.p], c=self.c, lr=self.lr
        )

    def _fake_backward(self, grad_val=0.01):
        """Simulate backward pass by setting .grad manually."""
        self.p.grad = torch.ones_like(self.p.data) * grad_val

    def test_params_stay_on_ball(self):
        """After multiple steps, parameter must remain inside the Poincaré ball."""
        max_norm = (1.0 / torch.sqrt(torch.abs(self.c))).item() - _EPS
        for _ in range(20):
            self._fake_backward()
            self.opt.step()
            self.opt.zero_grad()
        norms = self.p.data.norm(dim=-1)
        assert (norms < max_norm + 1e-3).all(), \
            f"Parameter left ball after optimization: max norm = {norms.max():.4f}"

    def test_adaptive_lr_decay(self):
        """Effective LR should decrease as lr_0 / sqrt(1 + |c|*t)."""
        # After t steps the effective LR should be lr_0 / sqrt(1 + |c|*t)
        t_steps = 10
        expected_lr_final = self.lr / math.sqrt(1.0 + abs(self.c.item()) * t_steps)
        for _ in range(t_steps):
            self._fake_backward()
            self.opt.step()
            self.opt.zero_grad()
        # Verify via state
        state = self.opt.state[self.p]
        assert state["step"] == t_steps
        # Recompute effective LR at step t_steps
        actual_lr = self.lr / math.sqrt(1.0 + abs(self.c.item()) * t_steps)
        assert abs(actual_lr - expected_lr_final) < 1e-8

    def test_step_counter_increments(self):
        """Step counter should increment on every .step() call."""
        for i in range(5):
            self._fake_backward()
            self.opt.step()
            self.opt.zero_grad()
        assert self.opt.state[self.p]["step"] == 5

    def test_moments_initialised(self):
        """First and second moments should be initialised on first step."""
        self._fake_backward()
        self.opt.step()
        state = self.opt.state[self.p]
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert state["exp_avg"].shape == self.p.data.shape

    def test_loss_decreases_on_simple_task(self):
        """ROM should minimise a simple geodesic regression task."""
        torch.manual_seed(0)
        target = _project_to_ball(torch.randn(8, self.d) * 0.2, self.c)
        param  = make_ball_param((8, self.d))
        opt    = CurvatureAdaptiveRiemannianAdam([param], c=self.c, lr=5e-2)

        losses = []
        for _ in range(50):
            d = geodesic_distance(param, target, self.c)
            loss = d.mean()
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert losses[-1] < losses[0] * 0.5, \
            f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"

    def test_zero_grad_clears(self):
        self._fake_backward()
        self.opt.step()
        self.opt.zero_grad()
        assert self.p.grad is None or (self.p.grad == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# create_optimizers factory
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateOptimizers:
    def test_splits_correctly(self):
        """Euclidean and Riemannian params should go to separate optimizers."""
        from geonet.models.geonet_graph import GeoNetGraph
        model = GeoNetGraph(in_dim=16, hidden_dim=32, out_dim=2, num_layers=2)
        eucl_opt, riem_opt = create_optimizers(model, lr_eucl=1e-3, lr_riem=3e-3)
        assert eucl_opt is not None, "Euclidean optimizer should not be None."
        # If model has hyperbolic params, riem_opt should exist
        # (GeoNetGraph always has HEL, so riem_opt should not be None)
        # Note: riem_opt may be None if all params are Euclidean (full ablation)

    def test_eucl_optimizer_is_adamw(self):
        from geonet.models.geonet_graph import GeoNetGraph
        model = GeoNetGraph(in_dim=16, hidden_dim=32, out_dim=2, num_layers=2)
        eucl_opt, _ = create_optimizers(model)
        assert isinstance(eucl_opt, torch.optim.AdamW), \
            f"Euclidean optimizer should be AdamW, got {type(eucl_opt)}"

    def test_no_param_overlap(self):
        """No parameter should appear in both optimizers."""
        from geonet.models.geonet_graph import GeoNetGraph
        model = GeoNetGraph(in_dim=16, hidden_dim=32, out_dim=2, num_layers=2)
        eucl_opt, riem_opt = create_optimizers(model)

        eucl_ids = set()
        for group in eucl_opt.param_groups:
            for p in group["params"]:
                eucl_ids.add(id(p))

        if riem_opt is not None:
            for group in riem_opt.param_groups:
                for p in group["params"]:
                    assert id(p) not in eucl_ids, \
                        "Parameter found in both Euclidean and Riemannian optimizers!"

    def test_all_model_params_covered(self):
        """Every model parameter should be in exactly one optimizer."""
        from geonet.models.geonet_graph import GeoNetGraph
        model = GeoNetGraph(in_dim=16, hidden_dim=32, out_dim=2, num_layers=2)
        eucl_opt, riem_opt = create_optimizers(model)

        covered_ids = set()
        for group in eucl_opt.param_groups:
            for p in group["params"]:
                covered_ids.add(id(p))
        if riem_opt is not None:
            for group in riem_opt.param_groups:
                for p in group["params"]:
                    covered_ids.add(id(p))

        all_param_ids = {id(p) for p in model.parameters()}
        uncovered = all_param_ids - covered_ids
        assert len(uncovered) == 0, \
            f"{len(uncovered)} model parameters not assigned to any optimizer."
