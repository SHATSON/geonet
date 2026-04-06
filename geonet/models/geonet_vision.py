"""
geonet/models/geonet_vision.py
───────────────────────────────
GeoNet vision model — used for CIFAR-100 and ImageNet-1K (paper Section 7.3).

Architecture:
  Backbone  : ResNet-101 feature extractor (removes the final FC layer)
  Projector : HEL — map CNN features onto the Poincaré ball
  Processor : Stack of GAA layers over patch tokens (ViT-style sequence)
  Decoder   : log_0 projection → hierarchical classification head

The hierarchical label structure (CIFAR-100: 100 classes / 20 superclasses;
ImageNet: 1000 classes in WordNet hierarchy) motivates hyperbolic embeddings
for improved HP@5 scores (Section 7.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from geonet.layers.hyperbolic_embedding import HyperbolicEmbeddingLayer
from geonet.layers.activations import TangentSpaceActivation, HyperbolicLayerNorm, HyperbolicDropout
from geonet.attention.geodesic_attention import GeometryAwareAttention
from geonet.utils.manifold import log_map_zero, exp_map_zero


class GeoNetVision(nn.Module):
    """GeoNet image classifier with hierarchical-label-aware embeddings.

    Parameters
    ----------
    num_classes   : int   — number of fine-grained classes (100 or 1000)
    num_superclass: int   — number of superclasses (20 for CIFAR-100, 0 = disabled)
    hidden_dim    : int   — GeoNet hyperbolic dimension (paper: 64)
    num_layers    : int   — GAA processor layers (paper: 3)
    num_heads     : int   — attention heads (paper: 4)
    c_init        : float — initial curvature
    learn_c       : bool
    dropout       : float
    backbone      : str   — 'resnet101' | 'resnet50'
    pretrained    : bool  — use ImageNet pre-trained backbone weights
    """

    # ResNet output feature dimensions
    _BACKBONE_DIMS = {"resnet101": 2048, "resnet50": 2048}

    def __init__(
        self,
        num_classes: int = 100,
        num_superclass: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        c_init: float = -1.0,
        learn_c: bool = True,
        dropout: float = 0.1,
        backbone: str = "resnet101",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes    = num_classes
        self.num_superclass = num_superclass

        # ── CNN Backbone ──────────────────────────────────────────────────────
        cnn = getattr(tvm, backbone)(
            weights=tvm.ResNet101_Weights.DEFAULT if pretrained and backbone == "resnet101"
            else tvm.ResNet50_Weights.DEFAULT if pretrained
            else None
        )
        # Remove final FC layer; keep everything up to global avg pool
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        cnn_dim  = self._BACKBONE_DIMS[backbone]

        # ── Hyperbolic projector ──────────────────────────────────────────────
        self.encoder = HyperbolicEmbeddingLayer(
            cnn_dim, hidden_dim, c_init=c_init, learn_c=learn_c, dropout=dropout
        )

        # ── GAA processor (applied over a single-token "sequence") ────────────
        self.gaa_layers  = nn.ModuleList([
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

        # ── Classification head ───────────────────────────────────────────────
        self.fine_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Optional auxiliary superclass head (improves hierarchical precision)
        if num_superclass > 0:
            self.super_head = nn.Sequential(
                nn.Linear(hidden_dim, num_superclass),
            )
        else:
            self.super_head = None

    @property
    def c(self) -> torch.Tensor:
        return self.encoder.c

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images : Tensor (B, 3, H, W)

        Returns
        -------
        dict with keys:
          'fine_logits'  : Tensor (B, num_classes)
          'super_logits' : Tensor (B, num_superclass)  (if num_superclass > 0)
          'embeddings'   : Tensor (B, hidden_dim)      — hyperbolic (for distortion eval)
        """
        c = self.c

        # ── CNN features ──────────────────────────────────────────────────────
        feat = self.cnn(images)                              # (B, 2048, 1, 1)
        feat = feat.flatten(1)                               # (B, 2048)

        # ── Hyperbolic projection ─────────────────────────────────────────────
        h = self.encoder(feat)                               # (B, d)

        # ── GAA processor over the single global token ────────────────────────
        h_seq = h.unsqueeze(1)                               # (B, 1, d)
        for gaa, norm, drop in zip(self.gaa_layers, self.norm_layers, self.drop_layers):
            h_tan = log_map_zero(h_seq, c)
            h_attn, _ = gaa(h_seq, h_seq, h_seq)
            h_attn_tan = log_map_zero(h_attn, c)
            h_seq = exp_map_zero(h_tan + h_attn_tan, c)
            h_seq = norm(h_seq, c)
            h_seq = drop(h_seq, c)
        h = h_seq.squeeze(1)                                 # (B, d)

        # ── Decode to Euclidean for classification ────────────────────────────
        h_eucl = log_map_zero(h, c)                          # (B, d)
        fine_logits = self.fine_head(h_eucl)                 # (B, C)

        result = {
            "fine_logits": fine_logits,
            "embeddings":  h,                                # hyperbolic, for metrics
        }
        if self.super_head is not None:
            result["super_logits"] = self.super_head(h_eucl)

        return result
