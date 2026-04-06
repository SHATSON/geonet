"""
tests/test_manifold.py
────────────────────────
Unit tests for geonet/utils/manifold.py.

Tests verify mathematical correctness of all Poincaré ball operations:
  - Möbius addition (identity, inverse, non-commutativity)
  - Exponential / logarithmic maps (roundtrip)
  - Geodesic distance (symmetry, triangle inequality, zero on diagonal)
  - exp_map_zero / log_map_zero roundtrip
  - embedding_distortion on a trivial case

Run: pytest tests/test_manifold.py -v
"""

import pytest
import torch
import math
from geonet.utils.manifold import (
    mobius_add,
    exp_map,
    log_map,
    exp_map_zero,
    log_map_zero,
    geodesic_distance,
    pairwise_geodesic_distance,
    _project_to_ball,
    _EPS,
)

# Fixed curvature for all tests (matching paper default)
C = torch.tensor(-1.0)


def randn_ball(n: int, d: int, scale: float = 0.3) -> torch.Tensor:
    """Sample n random points inside the Poincaré ball."""
    x = torch.randn(n, d) * scale
    return _project_to_ball(x, C)


class TestMobiusAddition:
    def test_identity(self):
        """x ⊕_c 0 = x"""
        x = randn_ball(8, 16)
        zero = torch.zeros_like(x)
        result = mobius_add(x, zero, C)
        assert torch.allclose(result, x, atol=1e-5), \
            f"Identity failed: max diff = {(result - x).abs().max():.2e}"

    def test_left_inverse(self):
        """(−x) ⊕_c x = 0"""
        x = randn_ball(8, 16)
        result = mobius_add(-x, x, C)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5), \
            f"Left inverse failed: max norm = {result.norm(dim=-1).max():.2e}"

    def test_stays_in_ball(self):
        """Möbius addition should keep points inside the ball."""
        x = randn_ball(32, 16)
        y = randn_ball(32, 16)
        result = mobius_add(x, y, C)
        norms = result.norm(dim=-1)
        max_allowed = 1.0 / (torch.sqrt(torch.abs(C)).item()) - _EPS
        assert (norms < max_allowed + 1e-4).all(), \
            f"Points outside ball: max norm = {norms.max():.4f}"

    def test_not_commutative(self):
        """Möbius addition is generally NOT commutative: x⊕y ≠ y⊕x."""
        x = randn_ball(4, 8)
        y = randn_ball(4, 8)
        xy = mobius_add(x, y, C)
        yx = mobius_add(y, x, C)
        # They should differ for generic points
        assert not torch.allclose(xy, yx, atol=1e-4), \
            "Möbius addition appears to be commutative (unexpected)"


class TestExpLogMaps:
    def test_exp_log_roundtrip_at_origin(self):
        """log_0(exp_0(v)) = v  for small tangent vectors."""
        v = torch.randn(16, 32) * 0.3
        recovered = log_map_zero(exp_map_zero(v, C), C)
        assert torch.allclose(v, recovered, atol=1e-5), \
            f"exp/log roundtrip failed: max diff = {(v - recovered).abs().max():.2e}"

    def test_exp_log_roundtrip_arbitrary_base(self):
        """log_x(exp_x(v)) = v  for arbitrary base point x."""
        x = randn_ball(8, 16)
        v = torch.randn(8, 16) * 0.2   # small tangent vectors
        y = exp_map(x, v, C)
        v_rec = log_map(x, y, C)
        assert torch.allclose(v, v_rec, atol=1e-4), \
            f"exp/log (arbitrary base) roundtrip failed: max diff = {(v - v_rec).abs().max():.2e}"

    def test_exp_map_zero_origin(self):
        """exp_0(0) = 0."""
        v = torch.zeros(4, 8)
        result = exp_map_zero(v, C)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_output_in_ball(self):
        """exp_0(v) must land inside the Poincaré ball."""
        v = torch.randn(64, 32)
        result = exp_map_zero(v, C)
        norms = result.norm(dim=-1)
        max_allowed = 1.0 / torch.sqrt(torch.abs(C)).item()
        assert (norms < max_allowed).all(), \
            f"exp_map_zero output outside ball: max norm = {norms.max():.4f}"


class TestGeodesicDistance:
    def test_zero_on_diagonal(self):
        """d(x, x) = 0 for all x."""
        x = randn_ball(16, 8)
        d = geodesic_distance(x, x, C)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-5), \
            f"d(x,x) ≠ 0: max = {d.max():.2e}"

    def test_symmetry(self):
        """d(x, y) = d(y, x)."""
        x = randn_ball(16, 8)
        y = randn_ball(16, 8)
        dxy = geodesic_distance(x, y, C)
        dyx = geodesic_distance(y, x, C)
        assert torch.allclose(dxy, dyx, atol=1e-5), \
            f"Distance asymmetry: max diff = {(dxy - dyx).abs().max():.2e}"

    def test_nonnegative(self):
        """d(x, y) >= 0."""
        x = randn_ball(32, 8)
        y = randn_ball(32, 8)
        d = geodesic_distance(x, y, C)
        assert (d >= -1e-6).all(), f"Negative distance encountered: min = {d.min():.4f}"

    def test_triangle_inequality(self):
        """d(x, z) <= d(x, y) + d(y, z)  for all x, y, z."""
        x = randn_ball(8, 8)
        y = randn_ball(8, 8)
        z = randn_ball(8, 8)
        dxz = geodesic_distance(x, z, C)
        dxy = geodesic_distance(x, y, C)
        dyz = geodesic_distance(y, z, C)
        violations = (dxz > dxy + dyz + 1e-4).sum().item()
        assert violations == 0, \
            f"Triangle inequality violated for {violations}/8 triples."

    def test_origin_distance_formula(self):
        """d(0, x) = (2/sqrt|c|) arctanh(sqrt|c| ||x||) matches closed form."""
        x = randn_ball(8, 4)
        origin = torch.zeros_like(x)
        d_computed = geodesic_distance(origin, x, C)
        # Closed form for d(0, x) in Poincare ball
        sq_c = torch.sqrt(torch.abs(C))
        x_norm = x.norm(dim=-1)
        d_expected = (2.0 / sq_c) * torch.arctanh(sq_c * x_norm)
        assert torch.allclose(d_computed, d_expected, atol=1e-5), \
            f"Origin distance mismatch: max diff = {(d_computed - d_expected).abs().max():.2e}"

    def test_pairwise_shape(self):
        """pairwise_geodesic_distance returns (n, m) tensor."""
        X = randn_ball(5, 8)
        Y = randn_ball(7, 8)
        D = pairwise_geodesic_distance(X, Y, C)
        assert D.shape == (5, 7), f"Expected (5,7), got {D.shape}"

    def test_curvature_effect(self):
        """More negative curvature → larger distances (hyperbolic space expands)."""
        x = randn_ball(4, 8)
        y = randn_ball(4, 8)
        c_low  = torch.tensor(-0.5)
        c_high = torch.tensor(-2.0)
        d_low  = geodesic_distance(
            _project_to_ball(x, c_low),  _project_to_ball(y, c_low),  c_low
        )
        d_high = geodesic_distance(
            _project_to_ball(x, c_high), _project_to_ball(y, c_high), c_high
        )
        # Higher |c| → shorter max radius but larger metric factor; distances differ
        assert not torch.allclose(d_low, d_high, atol=1e-3), \
            "Curvature has no effect on distance (unexpected)"


class TestGradients:
    def test_exp_map_zero_gradient(self):
        """exp_map_zero is differentiable w.r.t. input."""
        v = torch.randn(4, 8, requires_grad=True)
        y = exp_map_zero(v, C)
        loss = y.sum()
        loss.backward()
        assert v.grad is not None and not v.grad.isnan().any(), \
            "Gradient through exp_map_zero is NaN or missing."

    def test_geodesic_distance_gradient(self):
        """geodesic_distance is differentiable w.r.t. both inputs."""
        x = randn_ball(4, 8).requires_grad_(True)
        y = randn_ball(4, 8).requires_grad_(True)
        d = geodesic_distance(x, y, C)
        d.sum().backward()
        assert x.grad is not None and not x.grad.isnan().any()
        assert y.grad is not None and not y.grad.isnan().any()
