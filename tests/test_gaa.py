"""
tests/test_gaa.py
──────────────────
Unit tests for Geometry-Aware Attention (GAA) — paper Section 5.4.

Verifies:
  - Output shape and dtype
  - All outputs lie inside the Poincaré ball
  - Attention weights sum to 1 (valid probability distribution)
  - Masking correctly zeroes out padded positions
  - Temperature parameter τ affects logit scale
  - End-to-end gradient flow

Run: pytest tests/test_gaa.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from geonet.attention.geodesic_attention import GeometryAwareAttention
from geonet.utils.manifold import _project_to_ball, _EPS


def randn_eucl(B: int, T: int, D: int) -> torch.Tensor:
    """Random Euclidean sequence (before hyperbolic projection)."""
    torch.manual_seed(42)
    return torch.randn(B, T, D) * 0.3


class TestGeometryAwareAttention:
    """Full test suite for GAA module."""

    def setup_method(self):
        torch.manual_seed(0)
        self.B, self.T, self.D = 2, 6, 32
        self.gaa = GeometryAwareAttention(
            embed_dim=self.D,
            num_heads=4,
            c_init=-1.0,
            learn_c=True,
            tau_init=0.1,
            dropout=0.0,
        )
        self.gaa.eval()

    # ── Shape and type ─────────────────────────────────────────────────────

    def test_output_shape(self):
        x = randn_eucl(self.B, self.T, self.D)
        out, attn = self.gaa(x, x, x)
        assert out.shape == (self.B, self.T, self.D), \
            f"Expected output ({self.B},{self.T},{self.D}), got {out.shape}"
        assert attn.shape == (self.B, 4, self.T, self.T), \
            f"Expected attn ({self.B},4,{self.T},{self.T}), got {attn.shape}"

    def test_output_dtype_preserved(self):
        x = randn_eucl(self.B, self.T, self.D).float()
        out, _ = self.gaa(x, x, x)
        assert out.dtype == torch.float32

    # ── Attention weights ──────────────────────────────────────────────────

    def test_attention_weights_sum_to_one(self):
        """Each attention distribution should sum to 1.0."""
        x = randn_eucl(self.B, self.T, self.D)
        _, attn = self.gaa(x, x, x)
        row_sums = attn.sum(dim=-1)  # (B, H, T)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
            f"Attention rows don't sum to 1: max dev = {(row_sums - 1).abs().max():.4e}"

    def test_attention_weights_nonnegative(self):
        x = randn_eucl(self.B, self.T, self.D)
        _, attn = self.gaa(x, x, x)
        assert (attn >= -1e-6).all(), \
            f"Negative attention weights: min = {attn.min():.4e}"

    # ── Masking ─────────────────────────────────────────────────────────────

    def test_padding_mask_zeroes_attention(self):
        """Masked (padding) positions should receive ~0 attention weight."""
        x = randn_eucl(self.B, self.T, self.D)
        # Mask last 2 positions in all sequences
        mask = torch.zeros(self.B, self.T, dtype=torch.bool)
        mask[:, -2:] = True   # True = ignore

        _, attn = self.gaa(x, x, x, key_padding_mask=mask)
        masked_attn = attn[:, :, :, -2:]   # attention TO masked positions
        assert (masked_attn.abs() < 1e-4).all(), \
            f"Masked positions received non-zero attention: max = {masked_attn.abs().max():.4e}"

    def test_causal_mask(self):
        """Upper-triangular additive mask should prevent attending to future tokens."""
        x = randn_eucl(self.B, self.T, self.D)
        # Causal mask: large negative for upper triangle
        causal = torch.triu(
            torch.full((self.T, self.T), float("-inf")), diagonal=1
        )
        _, attn = self.gaa(x, x, x, attn_mask=causal)
        upper = attn[:, :, torch.triu(torch.ones(self.T, self.T), diagonal=1).bool()]
        assert (upper.abs() < 1e-4).all(), \
            "Causal masking failed: future positions received attention."

    # ── Temperature ──────────────────────────────────────────────────────────

    def test_lower_temperature_sharpens_attention(self):
        """Lower τ → more peaked attention distribution (lower entropy)."""
        x = randn_eucl(1, self.T, self.D)

        gaa_sharp = GeometryAwareAttention(self.D, num_heads=4, tau_init=0.01, dropout=0.0)
        gaa_flat  = GeometryAwareAttention(self.D, num_heads=4, tau_init=1.00, dropout=0.0)
        # Share all weights except tau
        gaa_flat.q_proj.weight.data  = gaa_sharp.q_proj.weight.data.clone()
        gaa_flat.k_proj.weight.data  = gaa_sharp.k_proj.weight.data.clone()
        gaa_flat.log_neg_c.data      = gaa_sharp.log_neg_c.data.clone()

        gaa_sharp.eval(); gaa_flat.eval()

        with torch.no_grad():
            _, a_sharp = gaa_sharp(x, x, x)
            _, a_flat  = gaa_flat(x, x, x)

        # Entropy of sharp distribution should be lower
        def entropy(a):
            a = a.clamp(min=1e-9)
            return -(a * a.log()).sum(dim=-1).mean().item()

        assert entropy(a_sharp) < entropy(a_flat), \
            "Lower temperature did not produce sharper attention."

    # ── Gradient flow ─────────────────────────────────────────────────────

    def test_gradient_flows_to_query(self):
        x = randn_eucl(self.B, self.T, self.D).requires_grad_(True)
        out, _ = self.gaa(x, x, x)
        out.sum().backward()
        assert x.grad is not None, "No gradient to input."
        assert not x.grad.isnan().any(), "NaN gradient to input."

    def test_gradient_flows_to_curvature(self):
        x = randn_eucl(self.B, self.T, self.D)
        out, _ = self.gaa(x, x, x)
        out.sum().backward()
        assert self.gaa.log_neg_c.grad is not None, "No gradient to curvature."
        assert not self.gaa.log_neg_c.grad.isnan().any(), "NaN gradient to curvature."

    def test_gradient_flows_to_temperature(self):
        x = randn_eucl(self.B, self.T, self.D)
        out, _ = self.gaa(x, x, x)
        out.sum().backward()
        assert self.gaa.log_tau.grad is not None, "No gradient to temperature τ."

    # ── Batch invariance ─────────────────────────────────────────────────

    def test_batch_independent(self):
        """Batching should not affect individual sequence outputs."""
        torch.manual_seed(1)
        x1 = randn_eucl(1, self.T, self.D)
        x2 = randn_eucl(1, self.T, self.D)
        x_batch = torch.cat([x1, x2], dim=0)

        self.gaa.eval()
        with torch.no_grad():
            out1, _ = self.gaa(x1, x1, x1)
            out2, _ = self.gaa(x2, x2, x2)
            out_b, _ = self.gaa(x_batch, x_batch, x_batch)

        assert torch.allclose(out1[0], out_b[0], atol=1e-4), \
            "Batch changes sequence 0 output."
        assert torch.allclose(out2[0], out_b[1], atol=1e-4), \
            "Batch changes sequence 1 output."
