"""
geonet/layers/activations.py
──────────────────────────────
Tangent-space activation functions for hyperbolic networks.

Implements the Gao et al. (2023) strategy referenced in Section 5.2:
    act_h(x) = exp_0^c( σ( log_0^c(x) ) )

Activations are applied in Euclidean tangent space to avoid the numerical
instabilities of directly applying nonlinearities on the manifold surface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geonet.utils.manifold import exp_map_zero, log_map_zero


class TangentSpaceActivation(nn.Module):
    """Apply a standard nonlinearity via tangent-space projection.

    Parameters
    ----------
    activation : str  — 'relu' | 'gelu' | 'tanh' | 'silu'
    """

    _ACTIVATIONS = {
        "relu": F.relu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
        "silu": F.silu,
        "leaky_relu": F.leaky_relu,
    }

    def __init__(self, activation: str = "relu") -> None:
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from: {list(self._ACTIVATIONS)}"
            )
        self._fn = self._ACTIVATIONS[activation]
        self.activation_name = activation

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (..., d)  — points on the Poincaré ball
        c : Tensor ()        — curvature

        Returns
        -------
        Tensor (..., d)  — points on the Poincaré ball after activation
        """
        v = log_map_zero(x, c)       # project to tangent space
        v = self._fn(v)              # apply activation in Euclidean tangent space
        return exp_map_zero(v, c)    # project back to manifold

    def extra_repr(self) -> str:
        return f"activation={self.activation_name}"


class HyperbolicDropout(nn.Module):
    """Dropout in tangent space (Gao et al., 2023).

    Applies standard Bernoulli dropout to tangent-space vectors and re-projects
    to the manifold.  More geometrically consistent than masking manifold points.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p
        self._dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        v = log_map_zero(x, c)
        v = self._dropout(v)
        return exp_map_zero(v, c)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class HyperbolicLayerNorm(nn.Module):
    """Layer normalisation in tangent space.

    Applies standard LayerNorm to the tangent-space projection of x.
    This preserves the approximate Euclidean structure near the origin,
    which is sufficient for the GeoNet architecture where embeddings
    remain in a moderate-norm regime.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v = log_map_zero(x, c)
        v = self.norm(v)
        return exp_map_zero(v, c)
