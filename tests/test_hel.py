"""
tests/test_hel.py
──────────────────
Unit tests for Hyperbolic Embedding Layer (HEL).

Run: pytest tests/test_hel.py -v
"""
import pytest
import torch
from geonet.layers.hyperbolic_embedding import HyperbolicEmbeddingLayer, HyperbolicMLPLayer
from geonet.utils.manifold import _project_to_ball, _EPS


C_DEFAULT = torch.tensor(-1.0)


class TestHyperbolicEmbeddingLayer:
    def setup_method(self):
        torch.manual_seed(0)
        self.layer = HyperbolicEmbeddingLayer(in_dim=32, out_dim=64, c_init=-1.0, learn_c=True)

    def test_output_shape(self):
        x = torch.randn(8, 32)
        y = self.layer(x)
        assert y.shape == (8, 64), f"Expected (8,64), got {y.shape}"

    def test_output_inside_ball(self):
        """All output points must lie strictly inside the Poincaré ball."""
        x = torch.randn(32, 32)
        y = self.layer(x)
        max_norm = (1.0 / torch.sqrt(torch.abs(self.layer.c))).item() - _EPS
        norms = y.norm(dim=-1)
        assert (norms < max_norm + 1e-3).all(), \
            f"HEL output outside ball: max norm = {norms.max():.4f}, max_allowed = {max_norm:.4f}"

    def test_learnable_curvature(self):
        """Curvature parameter should be updated by gradient."""
        x = torch.randn(4, 32)
        y = self.layer(x)
        loss = y.sum()
        loss.backward()
        assert self.layer.log_neg_c.grad is not None, "Curvature has no gradient."

    def test_fixed_curvature_no_grad(self):
        """With learn_c=False, curvature should have no gradient."""
        layer = HyperbolicEmbeddingLayer(32, 64, c_init=-1.0, learn_c=False)
        x = torch.randn(4, 32)
        y = layer(x)
        y.sum().backward()
        assert not isinstance(layer.log_neg_c, torch.nn.Parameter), \
            "Fixed curvature should not be a Parameter."

    def test_batch_dimensions(self):
        """HEL should handle arbitrary batch dimensions."""
        x = torch.randn(2, 4, 32)
        y = self.layer(x)
        assert y.shape == (2, 4, 64)

    def test_curvature_clamped(self):
        """Curvature should always stay within [-5, -0.01]."""
        c = self.layer.c
        assert -5.01 <= c.item() <= -0.009, \
            f"Curvature out of bounds: {c.item():.4f}"

    def test_dropout_training_vs_eval(self):
        """Dropout should only apply in training mode."""
        layer = HyperbolicEmbeddingLayer(32, 64, c_init=-1.0, dropout=0.9)
        x = torch.randn(64, 32)
        layer.eval()
        with torch.no_grad():
            y_eval1 = layer(x)
            y_eval2 = layer(x)
        assert torch.allclose(y_eval1, y_eval2), "Eval mode outputs are non-deterministic."


class TestHyperbolicMLPLayer:
    def test_output_shape(self):
        mlp = HyperbolicMLPLayer(dim=64, num_layers=2)
        x = _project_to_ball(torch.randn(8, 64) * 0.3, C_DEFAULT)
        y = mlp(x)
        assert y.shape == (8, 64)

    def test_output_in_ball(self):
        mlp = HyperbolicMLPLayer(dim=32, num_layers=2)
        x = _project_to_ball(torch.randn(16, 32) * 0.3, C_DEFAULT)
        y = mlp(x)
        norms = y.norm(dim=-1)
        max_norm = 1.0 / torch.sqrt(torch.abs(mlp.c)).item()
        assert (norms < max_norm + 1e-3).all()
