"""
geonet/layers/hyperbolic_linear.py
────────────────────────────────────
Hyperbolic linear map using the Möbius transformation formalism.

Implements the between-layer transform described in Section 5.2:
    M_h(x; W, b) = exp_0^c( W · log_0^c(x) + b )

This projects to tangent space, applies a Euclidean linear transform,
then projects back to the manifold — enabling standard backpropagation
while respecting the manifold structure.

References
----------
Ganea et al. (2018) NeurIPS; Chami et al. (2022) NeurIPS.
"""

import torch
import torch.nn as nn

from geonet.utils.manifold import exp_map_zero, log_map_zero, _clamp_curvature


class HyperbolicLinear(nn.Module):
    """Hyperbolic linear layer  M_h(x; W, b).

    Parameters
    ----------
    in_dim  : int
    out_dim : int
    c       : nn.Parameter or Tensor — shared curvature from parent module
    bias    : bool
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias   = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (..., in_dim)  — points on the Poincaré ball
        c : Tensor ()             — curvature

        Returns
        -------
        Tensor (..., out_dim)  — points on the Poincaré ball
        """
        # Project to tangent space at origin
        v = log_map_zero(x, c)                     # (..., in_dim)

        # Euclidean linear transform
        out = v @ self.weight.T                     # (..., out_dim)
        if self.bias is not None:
            out = out + self.bias

        # Project back to manifold
        return exp_map_zero(out, c)


class HyperbolicGraphConv(nn.Module):
    """Hyperbolic graph convolution layer (HGCN-style, Chami et al., 2022).

    Aggregates neighbour features in tangent space at each node, then
    applies a hyperbolic linear transform.

    forward(x, edge_index, c):
        1. Project all x to tangent space at origin: v = log_0(x)
        2. Aggregate: v_agg[i] = sum_j A[i,j] * v[j]
        3. Transform: x_out = exp_0(W * v_agg + b)

    Parameters
    ----------
    in_dim  : int
    out_dim : int
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = HyperbolicLinear(in_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        c: torch.Tensor,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : Tensor (N, in_dim)  — hyperbolic node features
        edge_index : Tensor (2, E)       — source/target edge indices
        c          : Tensor ()           — curvature
        num_nodes  : int | None

        Returns
        -------
        Tensor (N, out_dim)
        """
        if num_nodes is None:
            num_nodes = x.size(0)

        # Project to tangent space
        v = log_map_zero(x, c)                  # (N, in_dim)

        # Aggregate: simple mean pooling over neighbours
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros(num_nodes, v.size(-1), dtype=v.dtype, device=v.device)
        count = torch.zeros(num_nodes, 1, dtype=v.dtype, device=v.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(v[src]), v[src])
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(len(dst), 1, device=v.device))
        # Self-loop
        agg = agg + v
        count = count + 1.0
        v_agg = agg / count.clamp(min=1.0)       # (N, in_dim)

        # Project back via hyperbolic linear
        v_hyp = exp_map_zero(v_agg, c)
        return self.linear(v_hyp, c)
