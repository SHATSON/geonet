"""
geonet/optim/riemannian_adam.py
────────────────────────────────
Riemannian Optimization Module (ROM) — paper Section 5.3.

Implements curvature-adaptive RiemannianAdam as described in Section 5.3,
grounded in Yang et al. (2024) AdaManifold and Sun et al. (2023) convergence
analysis.

Key differences from standard RiemannianAdam (Becigneul & Ganea, 2019):
1. Curvature-adaptive LR schedule:  lr_t = lr_0 / √(1 + |c| · t)
   — derived from Sun et al. (2023) O(1/√T) convergence bound.
2. Riemannian gradient projection:  grad_R = ((1 − |c|‖θ‖²)² / 4) · grad_E
   — the conformal-factor rescaling of Euclidean gradients.
3. Manifold retraction:  θ_{t+1} = exp_{θ_t}(−lr · m / (√v + ε))

Usage
-----
    from geonet.optim.riemannian_adam import create_optimizers

    eucl_opt, riem_opt = create_optimizers(model, lr_eucl=1e-3, lr_riem=3e-3)

    # Training loop
    eucl_opt.zero_grad(); riem_opt.zero_grad()
    loss.backward()
    eucl_opt.step(); riem_opt.step()

References
----------
Yang et al. (2024) ICLR; Sun et al. (2023) Math. Oper. Res.;
Becigneul & Ganea (2019) ICLR.
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Iterable, Optional, Tuple, List

from geonet.utils.manifold import exp_map, log_map, _clamp_curvature, _EPS


# ─────────────────────────────────────────────────────────────────────────────
# Riemannian gradient utilities
# ─────────────────────────────────────────────────────────────────────────────

def riemannian_gradient(
    eucl_grad: torch.Tensor,
    point: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Convert Euclidean gradient to Riemannian gradient on the Poincaré ball.

    grad_R = ((1 − |c|‖x‖²)² / 4) · grad_E

    This is the inverse of the conformal metric tensor applied to grad_E
    (Section 5.3; Becigneul & Ganea, 2019, eq. 3).
    """
    c = _clamp_curvature(c)
    x2 = (point * point).sum(dim=-1, keepdim=True)
    factor = ((1.0 - torch.abs(c) * x2) ** 2) / 4.0
    return factor * eucl_grad


# ─────────────────────────────────────────────────────────────────────────────
# CurvatureAdaptiveRiemannianAdam
# ─────────────────────────────────────────────────────────────────────────────

class CurvatureAdaptiveRiemannianAdam(Optimizer):
    """Riemannian Adam with curvature-adaptive learning rate schedule.

    Parameters
    ----------
    params  : parameter iterable  — Poincaré-ball parameters to optimise
    c       : torch.Tensor        — curvature scalar (shared, updated externally)
    lr      : float               — initial learning rate lr_0 (paper: 3e-3)
    betas   : Tuple[float, float] — Adam moment decay rates
    eps     : float               — numerical stability constant
    weight_decay : float          — L2 penalty coefficient

    Schedule
    --------
    lr_t = lr_0 / √(1 + |c| · t)   [Sun et al., 2023, Theorem 2]
    """

    def __init__(
        self,
        params: Iterable,
        c: torch.Tensor,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._c = c
        super().__init__(params, defaults)

    @property
    def c(self) -> torch.Tensor:
        return _clamp_curvature(self._c)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        c = self.c
        abs_c = torch.abs(c).item()

        for group in self.param_groups:
            lr0 = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd  = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Step counter
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"]    = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]

                # Curvature-adaptive learning rate (Sun et al., 2023)
                lr_t = lr0 / math.sqrt(1.0 + abs_c * t)

                # Euclidean gradient (with optional weight decay in tangent space)
                g_e = p.grad
                if wd != 0.0:
                    g_e = g_e + wd * p.data

                # Convert to Riemannian gradient
                g_r = riemannian_gradient(g_e, p.data, c)

                # Moment updates (in tangent space at current point)
                m  = state["exp_avg"]
                v  = state["exp_avg_sq"]
                m.mul_(beta1).add_(g_r, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_r, g_r, value=1.0 - beta2)

                # Bias correction
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                v_hat = v / bc2

                # Update direction in tangent space
                step_tan = -lr_t * m_hat / (v_hat.sqrt() + eps)  # (d,)

                # Retract back to manifold via exponential map
                p.data.copy_(exp_map(p.data, step_tan, c))

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Factory — used by train.py
# ─────────────────────────────────────────────────────────────────────────────

def _is_riemannian_param(name: str, module: nn.Module) -> bool:
    """Identify parameters that live on a Riemannian manifold.

    Parameters with 'log_neg_c' (curvature) are excluded — they are scalar
    scalars in R and updated by Euclidean Adam.  Only the actual hyperbolic
    embedding points need Riemannian update.
    """
    return "hel" in name.lower() or "hyperbolic" in name.lower()


def create_optimizers(
    model: nn.Module,
    lr_eucl: float = 1e-3,
    lr_riem: float = 3e-3,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> Tuple[torch.optim.Optimizer, Optional[CurvatureAdaptiveRiemannianAdam]]:
    """Split model parameters into Euclidean and Riemannian groups.

    All manifold-valued parameters (HEL outputs, GAA query/key projections
    on the Poincaré ball) are updated by CurvatureAdaptiveRiemannianAdam.
    All remaining parameters are updated by standard AdamW.

    Returns
    -------
    (eucl_optimizer, riem_optimizer)
    """
    eucl_params: List[torch.Tensor] = []
    riem_params: List[torch.Tensor] = []

    # Collect curvature tensor (shared across the model)
    c_tensor = None
    for name, param in model.named_parameters():
        if "log_neg_c" in name:
            eucl_params.append(param)   # curvature is a scalar in ℝ
            if c_tensor is None:
                # Reconstruct c from log_neg_c for the Riemannian optimizer
                c_tensor = -torch.exp(param.detach())
        elif "hel" in name or "hyperbolic" in name or "h_proj" in name:
            riem_params.append(param)
        else:
            eucl_params.append(param)

    eucl_opt = torch.optim.AdamW(
        eucl_params,
        lr=lr_eucl,
        betas=betas,
        weight_decay=weight_decay,
    )

    if riem_params and c_tensor is not None:
        riem_opt = CurvatureAdaptiveRiemannianAdam(
            riem_params,
            c=c_tensor,
            lr=lr_riem,
            betas=betas,
            weight_decay=weight_decay,
        )
    else:
        # No Riemannian parameters (e.g., fully ablated Euclidean model)
        riem_opt = None

    return eucl_opt, riem_opt
