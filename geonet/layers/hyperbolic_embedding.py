"""
geonet/layers/hyperbolic_embedding.py
──────────────────────────────────────
Hyperbolic Embedding Layer (HEL) — paper Section 5.2.

Maps Euclidean input features onto the Poincaré ball via:
    φ(x) = exp_0^c(Wx + b)

Curvature c is a learnable scalar parameter constrained to [-5, -0.01]
via a soft clipping function, initialised to -1.0 (paper Section 5.2).

References
----------
Ganea et al. (2018) NeurIPS; Chami et al. (2022) NeurIPS;
Gao et al. (2023) ICML (tangent-space activation strategy).
"""

import torch
import torch.nn as nn
from typing import Optional

from geonet.utils.manifold import exp_map_zero, _clamp_curvature, _project_to_ball


class HyperbolicEmbeddingLayer(nn.Module):
    """Map Euclidean vectors onto the Poincaré ball.

    Parameters
    ----------
    in_dim   : int    — input feature dimension
    out_dim  : int    — embedding dimension d (paper uses d=64)
    c_init   : float  — initial curvature (default -1.0 as in paper Section 5.2)
    learn_c  : bool   — whether to learn curvature (True → GeoNet; False →
                        GeoNet-fixed-c ablation)
    bias     : bool   — include bias term
    dropout  : float  — dropout rate applied in tangent space before projection

    Notes
    -----
    The Euclidean linear transform W,b lives in ordinary parameter space and
    is updated by the Euclidean Adam optimizer.  Only the curvature scalar c
    is a Riemannian parameter updated by ROM.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        c_init: float = -1.0,
        learn_c: bool = True,
        bias: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.learn_c = learn_c

        # Euclidean linear transform (updated by standard Adam)
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

        # Curvature parameter — stored as raw float, clamped in forward
        c_val = torch.tensor(float(c_init))
        if learn_c:
            self.log_neg_c = nn.Parameter(torch.log(-c_val))   # learn log|c|
        else:
            self.register_buffer("log_neg_c", torch.log(-c_val))

        self.dropout = nn.Dropout(p=dropout)

    @property
    def c(self) -> torch.Tensor:
        """Return curvature scalar, ensuring c ∈ [-5, -0.01]."""
        neg_c = torch.exp(self.log_neg_c)
        neg_c = torch.clamp(neg_c, min=0.01, max=5.0)
        return -neg_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (..., in_dim)  — Euclidean input features

        Returns
        -------
        Tensor (..., out_dim)  — points on the Poincaré ball
        """
        # Step 1: Euclidean linear transform
        v = self.linear(x)                   # (..., out_dim)
        v = self.dropout(v)                  # tangent-space dropout (Gao et al., 2023)

        # Step 2: Project onto manifold via exp_0
        return exp_map_zero(v, self.c)


class HyperbolicMLPLayer(nn.Module):
    """Multi-layer hyperbolic feature transform.

    Stacks HEL → tangent-space nonlinearity → HEL blocks as described in
    Section 5.2.  Shared curvature across all sub-layers by default.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        c_init: float = -1.0,
        learn_c: bool = True,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from geonet.layers.activations import TangentSpaceActivation
        self.layers = nn.ModuleList([
            HyperbolicEmbeddingLayer(dim, dim, c_init=c_init, learn_c=False, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.activations = nn.ModuleList([
            TangentSpaceActivation(activation=activation)
            for _ in range(num_layers - 1)
        ])
        # Shared learnable curvature
        neg_c = -float(c_init)
        if learn_c:
            self.log_neg_c = nn.Parameter(torch.log(torch.tensor(neg_c)))
        else:
            self.register_buffer("log_neg_c", torch.log(torch.tensor(neg_c)))

    @property
    def c(self) -> torch.Tensor:
        neg_c = torch.exp(self.log_neg_c).clamp(0.01, 5.0)
        return -neg_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Synchronise curvature across sub-layers at forward time
        c = self.c
        for layer in self.layers:
            layer.log_neg_c.data = self.log_neg_c.data
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x, c)
        return x
