"""
geonet/models/geonet_graph.py
──────────────────────────────
GeoNet graph model — paper Section 5, evaluated in Section 7.1.

Architecture (encoder → processor → decoder):
  Encoder  : HEL  (Euclidean features → Poincaré ball)
  Processor: stack of HyperbolicGraphConv + GAA + LayerNorm + Dropout
  Decoder  : log_0 projection → Euclidean linear head

Used for:
  - WordNet-Mammals link prediction (MAP, FD-accuracy, distortion)
  - ogbn-arxiv node classification (accuracy, distortion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geonet.layers.hyperbolic_embedding import HyperbolicEmbeddingLayer
from geonet.layers.hyperbolic_linear import HyperbolicGraphConv
from geonet.layers.activations import TangentSpaceActivation, HyperbolicLayerNorm, HyperbolicDropout
from geonet.attention.geodesic_attention import GeometryAwareAttention
from geonet.utils.manifold import log_map_zero, _clamp_curvature


class GeoNetGraph(nn.Module):
    """GeoNet for graph learning tasks.

    Parameters
    ----------
    in_dim      : int   — input node feature dimension
    hidden_dim  : int   — hidden / embedding dimension d (paper: 64)
    out_dim     : int   — number of output classes
    num_layers  : int   — number of GeoNet processor layers (paper: 3)
    num_heads   : int   — GAA attention heads (paper: 4)
    c_init      : float — initial curvature (paper: -1.0)
    learn_c     : bool  — learnable curvature
    dropout     : float — dropout rate (paper: 0.1)
    task        : str   — 'link' or 'node'
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 3,
        num_heads: int = 4,
        c_init: float = -1.0,
        learn_c: bool = True,
        dropout: float = 0.1,
        task: str = "node",
    ) -> None:
        super().__init__()
        self.task = task

        # ── Encoder: map node features onto the Poincaré ball ─────────────────
        self.encoder = HyperbolicEmbeddingLayer(
            in_dim, hidden_dim, c_init=c_init, learn_c=learn_c, dropout=dropout
        )

        # ── Processor: stack of geometry-aware graph layers ───────────────────
        self.conv_layers = nn.ModuleList([
            HyperbolicGraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.attn_layers = nn.ModuleList([
            GeometryAwareAttention(
                hidden_dim, num_heads=num_heads,
                c_init=c_init, learn_c=False, dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            HyperbolicLayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.drop_layers = nn.ModuleList([
            HyperbolicDropout(p=dropout) for _ in range(num_layers)
        ])
        self.act = TangentSpaceActivation(activation="relu")

        # ── Decoder: project to Euclidean, apply classification head ──────────
        if task == "node":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        elif task == "link":
            # For link prediction, score = f(d_H(u, v))
            self.link_head = nn.Linear(1, 1)  # learnable affine on distance

        # Shared curvature reference for inference
        self._c_init = c_init

    @property
    def c(self) -> torch.Tensor:
        return self.encoder.c

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode node features to hyperbolic embeddings.

        Parameters
        ----------
        x          : Tensor (N, in_dim)
        edge_index : Tensor (2, E)

        Returns
        -------
        Tensor (N, hidden_dim)  — points on the Poincaré ball
        """
        c = self.c
        h = self.encoder(x)                                          # (N, d)

        for conv, attn, norm, drop in zip(
            self.conv_layers, self.attn_layers, self.norm_layers, self.drop_layers
        ):
            # Graph convolution in hyperbolic space
            h_conv = conv(h, edge_index, c)                          # (N, d)
            # Self-attention over all nodes (batch = 1 sequence of N tokens)
            h_attn, _ = attn(
                h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0)
            )                                                        # (1, N, d)
            h_attn = h_attn.squeeze(0)                               # (N, d)
            # Residual: add in tangent space
            h_tan  = log_map_zero(h, c)
            hc_tan = log_map_zero(h_conv, c)
            ha_tan = log_map_zero(h_attn, c)
            from geonet.utils.manifold import exp_map_zero
            h = exp_map_zero(h_tan + hc_tan + ha_tan, c)            # residual
            h = norm(h, c)
            h = drop(h, c)
            h = self.act(h, c)

        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        src_idx: torch.Tensor | None = None,
        dst_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : Tensor (N, in_dim)
        edge_index : Tensor (2, E)
        src_idx    : Tensor (B,)  — source nodes for link prediction
        dst_idx    : Tensor (B,)  — target nodes for link prediction

        Returns
        -------
        For node task  : Tensor (N, out_dim) — class logits
        For link task  : Tensor (B,)         — link scores
        """
        h = self.encode(x, edge_index)        # (N, d)  hyperbolic
        c = self.c

        if self.task == "node":
            # Project to Euclidean tangent space for final classification
            h_eucl = log_map_zero(h, c)        # (N, d)
            return self.head(h_eucl)           # (N, out_dim)

        elif self.task == "link":
            assert src_idx is not None and dst_idx is not None
            from geonet.utils.manifold import geodesic_distance
            d = geodesic_distance(h[src_idx], h[dst_idx], c).unsqueeze(-1)
            return self.link_head(d).squeeze(-1)

        else:
            raise ValueError(f"Unknown task: {self.task!r}")
