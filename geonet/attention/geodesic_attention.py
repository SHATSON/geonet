"""
geonet/attention/geodesic_attention.py
───────────────────────────────────────
Geometry-Aware Attention (GAA) — paper Section 5.4.

Replaces the standard dot-product attention affinity with negative geodesic
distance in hyperbolic space:

    a_ij = −d_H^c(q_i, k_j) / τ

where d_H^c is the Poincaré-ball geodesic distance and τ is a learnable
temperature (initialised to 0.1, paper Section 5.4).

Output is the Möbius-weighted Fréchet mean of the value vectors.

Complexity: O(n² d) per layer — identical to standard attention.
The geodesic computation adds a constant factor of ≈2.3× relative to
dot-product attention (benchmarked on A100, Section 7.5).

References
----------
Section 5.4 of the paper; Ganea et al. (2018) NeurIPS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from geonet.utils.manifold import (
    geodesic_distance,
    pairwise_geodesic_distance,
    log_map_zero,
    exp_map_zero,
    frechet_mean,
    _clamp_curvature,
)
from geonet.layers.hyperbolic_embedding import HyperbolicEmbeddingLayer


class GeometryAwareAttention(nn.Module):
    """Multi-head Geometry-Aware Attention (GAA).

    Parameters
    ----------
    embed_dim   : int   — total embedding dimension (split across heads)
    num_heads   : int   — number of attention heads (paper uses 4)
    c_init      : float — initial curvature
    learn_c     : bool  — whether curvature is learnable
    tau_init    : float — initial attention temperature (paper: 0.1)
    dropout     : float — attention dropout rate

    Notes
    -----
    The four-head design of GeoNet replaces the eight-head configuration of
    comparable Euclidean transformers, reflecting the higher information density
    per head enabled by hyperbolic geometry (Section 5.4).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        c_init: float = -1.0,
        learn_c: bool = True,
        tau_init: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads

        # Learnable temperature τ (log-parameterised for positivity)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

        # Curvature (shared with the rest of GeoNet)
        neg_c = -float(c_init)
        if learn_c:
            self.log_neg_c = nn.Parameter(torch.log(torch.tensor(neg_c)))
        else:
            self.register_buffer("log_neg_c", torch.log(torch.tensor(neg_c)))

        # Projections for Q, K, V — operate in tangent space
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = HyperbolicEmbeddingLayer(
            embed_dim, embed_dim, c_init=c_init, learn_c=False, dropout=0.0
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(p=dropout)

        # Initialise projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)

    @property
    def c(self) -> torch.Tensor:
        return -torch.exp(self.log_neg_c).clamp(0.01, 5.0)

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(min=1e-3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query, key, value : Tensor (B, T, embed_dim)  — sequence inputs
            - query/key are Euclidean vectors (projected to tangent space here)
            - value should be points on the Poincaré ball
        key_padding_mask  : Tensor (B, T) bool, True = ignore position
        attn_mask         : Tensor (T, T) additive mask

        Returns
        -------
        output     : Tensor (B, T, embed_dim)  — hyperbolic attended values
        attn_probs : Tensor (B, heads, T, T)   — attention weights (for analysis)
        """
        B, T, _ = query.shape
        H, D = self.num_heads, self.head_dim
        c = self.c
        tau = self.tau

        # ── Project queries and keys (Euclidean tangent space) ────────────────
        # Shape after reshape: (B, T, H, D) → (B, H, T, D)
        Q = self.q_proj(query).view(B, T, H, D).transpose(1, 2)   # (B,H,T,D)
        K = self.k_proj(key).view(B, T, H, D).transpose(1, 2)     # (B,H,T,D)

        # Map Q, K onto the Poincaré ball for geodesic computation
        Q_hyp = exp_map_zero(Q.reshape(B * H * T, D), c).reshape(B, H, T, D)
        K_hyp = exp_map_zero(K.reshape(B * H * T, D), c).reshape(B, H, T, D)

        # ── Compute geodesic-distance attention logits ─────────────────────────
        # For each head, compute (T × T) pairwise distances
        # Q_hyp: (B,H,T,D) → flatten batch&heads → (B*H, T, D)
        Q_flat = Q_hyp.reshape(B * H, T, D)
        K_flat = K_hyp.reshape(B * H, T, D)

        # pairwise_geodesic_distance: (n, d), (m, d) → (n, m)
        # We need (B*H, T, T); loop over batch for memory efficiency
        dist_BH = torch.stack([
            pairwise_geodesic_distance(Q_flat[b], K_flat[b], c)   # (T, T)
            for b in range(B * H)
        ])  # (B*H, T, T)

        logits = -dist_BH / tau                                    # (B*H, T, T)
        logits = logits.reshape(B, H, T, T)

        # ── Apply masks ───────────────────────────────────────────────────────
        if attn_mask is not None:
            logits = logits + attn_mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            # (B, T) → (B, 1, 1, T)
            logits = logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_probs = F.softmax(logits, dim=-1)                     # (B, H, T, T)
        attn_probs = self.attn_dropout(attn_probs)

        # ── Project values to hyperbolic space ────────────────────────────────
        V_hyp = self.v_proj(value)                                 # (B, T, embed_dim)
        V_hyp = V_hyp.view(B, T, H, D).transpose(1, 2)            # (B, H, T, D)

        # ── Weighted aggregation: Möbius-weighted Fréchet mean per query ──────
        # For efficiency, use a tangent-space weighted sum approximation
        # (exact Fréchet mean is reserved for small T; see paper Section 5.4)
        # Tangent-space approximation: log → weighted sum → exp
        V_tan = log_map_zero(V_hyp.reshape(B * H * T, D), c).reshape(B, H, T, D)
        # attn_probs: (B, H, T, T), V_tan: (B, H, T, D)
        out_tan = torch.einsum("bhts,bhsd->bhtd", attn_probs, V_tan)  # (B, H, T, D)
        out_hyp = exp_map_zero(out_tan.reshape(B * H * T, D), c).reshape(B, H, T, D)

        # ── Merge heads ───────────────────────────────────────────────────────
        # Bring back to Euclidean for final linear projection
        out_tan_final = log_map_zero(out_hyp.reshape(B * H * T, D), c)
        out = out_tan_final.reshape(B, H, T, D).transpose(1, 2).reshape(B, T, embed_dim := H * D)
        out = self.out_proj(out)                                   # (B, T, embed_dim)

        return out, attn_probs
