"""
geonet/utils/manifold.py
────────────────────────
Poincaré ball manifold operations used throughout GeoNet.

All formulae follow the gyrovector space formalism of Ungar (2022) as adapted
by Ganea et al. (2018) and extended with learnable curvature per Chami et al.
(2022).  Every public function is differentiable end-to-end via PyTorch autograd.

Mathematical conventions
------------------------
- Curvature  c < 0  (negative).  |c| controls the "sharpness" of the ball.
- The Poincaré ball of curvature c is:
      B^n_c = { x ∈ ℝⁿ : |c| ‖x‖² < 1 }
- The conformal factor at x is:
      λ_c(x) = 2 / (1 − |c| ‖x‖²)
- Geodesic distance:
      d_c(x, y) = (2/√|c|) arctanh( √|c| ‖−x ⊕_c y‖ )
- Möbius addition:
      x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²) x + (1 − c‖x‖²) y)
                / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)

References
----------
Ungar (2022); Ganea et al. (2018) NeurIPS; Chami et al. (2022) NeurIPS.
"""

import torch
import torch.nn as nn
from typing import Optional

# Numerical stability clamp — keeps points strictly inside the ball
_EPS = 1e-5
_MAX_NORM_FACTOR = 1.0 - _EPS


# ─────────────────────────────────────────────────────────────────────────────
# Low-level scalar helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_curvature(c: torch.Tensor) -> torch.Tensor:
    """Ensure curvature is in [-5, -0.01] (paper Appendix C)."""
    return torch.clamp(c, min=-5.0, max=-0.01)


def _project_to_ball(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Project x onto the open Poincaré ball of curvature c.

    Applies a norm-clamp so ‖x‖ < 1/√|c|.
    """
    c = _clamp_curvature(c)
    max_norm = _MAX_NORM_FACTOR / (torch.sqrt(torch.abs(c)) + _EPS)
    norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=_EPS)
    # Only clamp points that violate the boundary
    clamped = x / norm * max_norm
    return torch.where(norm > max_norm, clamped, x)


def _lambda_x(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Conformal factor λ_c(x) = 2 / (1 − |c|‖x‖²).  Shape: (*batch, 1)."""
    c = _clamp_curvature(c)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - torch.abs(c) * x2).clamp(min=_EPS)


# ─────────────────────────────────────────────────────────────────────────────
# Core manifold operations
# ─────────────────────────────────────────────────────────────────────────────

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Möbius addition  x ⊕_c y  in the Poincaré ball.

    Parameters
    ----------
    x, y : Tensor  (..., d)  — points inside the ball
    c    : Tensor  ()        — curvature (scalar, c < 0)

    Returns
    -------
    Tensor (..., d)
    """
    c = _clamp_curvature(c)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denom = 1.0 + 2.0 * c * xy + c * c * x2 * y2
    return _project_to_ball(num / denom.clamp(min=_EPS), c)


def exp_map(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map  exp_x^c(v) : tangent vector v at x → point on ball.

    Parameters
    ----------
    x : Tensor (..., d)  — base point on the manifold
    v : Tensor (..., d)  — tangent vector at x
    c : Tensor ()        — curvature

    Returns
    -------
    Tensor (..., d) on the Poincaré ball.
    """
    c = _clamp_curvature(c)
    lam = _lambda_x(x, c)                             # (..., 1)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    arg = torch.tanh(torch.sqrt(torch.abs(c)) * lam * v_norm / 2.0)
    direction = v / v_norm
    y = arg / (torch.sqrt(torch.abs(c)) + _EPS) * direction
    return mobius_add(x, y, c)


def exp_map_zero(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map at the origin:  exp_0^c(v).

    Used in HEL to map Euclidean features onto the manifold (Section 5.2).
    """
    c = _clamp_curvature(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    arg = torch.tanh(torch.sqrt(torch.abs(c)) * v_norm)
    return _project_to_ball(arg / (torch.sqrt(torch.abs(c)) * v_norm + _EPS) * v, c)


def log_map(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Logarithmic map  log_x^c(y) : point y → tangent vector at x.

    Parameters
    ----------
    x : Tensor (..., d)  — base point
    y : Tensor (..., d)  — target point on the manifold
    c : Tensor ()        — curvature

    Returns
    -------
    Tensor (..., d)  tangent vector in T_x M.
    """
    c = _clamp_curvature(c)
    minus_x = -x
    add = mobius_add(minus_x, y, c)
    add_norm = add.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    lam = _lambda_x(x, c)
    arg = torch.sqrt(torch.abs(c)) * add_norm
    coeff = (2.0 / (torch.sqrt(torch.abs(c)) * lam + _EPS)) * torch.arctanh(arg.clamp(max=1.0 - _EPS))
    return coeff * add / add_norm


def log_map_zero(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Logarithmic map at the origin:  log_0^c(y)."""
    c = _clamp_curvature(c)
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    arg = torch.sqrt(torch.abs(c)) * y_norm
    coeff = (1.0 / (torch.sqrt(torch.abs(c)) + _EPS)) * torch.arctanh(arg.clamp(max=1.0 - _EPS))
    return coeff * y / y_norm


def geodesic_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Geodesic distance  d_c(x, y)  on the Poincaré ball.

    Parameters
    ----------
    x, y : Tensor (..., d)
    c    : Tensor ()  curvature

    Returns
    -------
    Tensor (...)  — pairwise distances.
    """
    c = _clamp_curvature(c)
    add = mobius_add(-x, y, c)
    add_norm = add.norm(dim=-1).clamp(min=_EPS)
    arg = torch.sqrt(torch.abs(c)) * add_norm
    return (2.0 / torch.sqrt(torch.abs(c))) * torch.arctanh(arg.clamp(max=1.0 - _EPS))


def pairwise_geodesic_distance(
    X: torch.Tensor, Y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Compute all pairwise geodesic distances between rows of X and Y.

    Parameters
    ----------
    X : Tensor (n, d)
    Y : Tensor (m, d)
    c : Tensor ()

    Returns
    -------
    Tensor (n, m)
    """
    # Expand for broadcasting: (n, 1, d) and (1, m, d)
    X_exp = X.unsqueeze(1)
    Y_exp = Y.unsqueeze(0)
    return geodesic_distance(X_exp, Y_exp, c)


def frechet_mean(
    points: torch.Tensor,
    weights: torch.Tensor,
    c: torch.Tensor,
    max_iter: int = 10,
) -> torch.Tensor:
    """Weighted Fréchet mean on the Poincaré ball via gradient descent.

    Parameters
    ----------
    points  : Tensor (n, d)   — points on the ball
    weights : Tensor (n,)     — non-negative weights (need not sum to 1)
    c       : Tensor ()       — curvature
    max_iter: int             — fixed-point iterations (10 is sufficient for
                                attention aggregation per Ganea et al., 2018)

    Returns
    -------
    Tensor (d,)  — weighted Fréchet mean.
    """
    weights = weights / (weights.sum() + _EPS)          # normalise
    mu = points[0].clone()                              # initialise at first point
    for _ in range(max_iter):
        logs = torch.stack([log_map(mu, p, c) for p in points], dim=0)  # (n, d)
        grad = (weights.unsqueeze(-1) * logs).sum(dim=0)                 # (d,)
        mu = exp_map(mu, grad, c)
    return mu


def embedding_distortion(
    embeddings: torch.Tensor,
    true_distances: torch.Tensor,
    c: torch.Tensor,
    n_samples: int = 10_000,
    seed: int = 42,
) -> float:
    """Estimate mean embedding distortion over sampled point pairs.

    distortion = E[ |d_emb(i,j) / d_true(i,j) − 1| ]

    Used for reporting Table 2 distortion values (Section 7.1).

    Parameters
    ----------
    embeddings     : Tensor (N, d)  — embedded points on the ball
    true_distances : Tensor (N, N)  — ground-truth pairwise distances
    c              : Tensor ()
    n_samples      : int
    seed           : int

    Returns
    -------
    float — mean distortion.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    N = embeddings.size(0)
    idx = torch.randint(0, N, (n_samples, 2), generator=rng)
    i, j = idx[:, 0], idx[:, 1]
    mask = i != j
    i, j = i[mask], j[mask]

    d_emb  = geodesic_distance(embeddings[i], embeddings[j], c).detach()
    d_true = true_distances[i, j]
    ratio  = d_emb / (d_true + _EPS)
    return (ratio - 1.0).abs().mean().item()
