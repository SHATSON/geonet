"""
geonet/models/geonet_nlp.py
────────────────────────────
GeoNet NLP model — used for SNLI and MultiNLI (paper Sections 5, 7.2).

Architecture:
  Backbone  : Pre-trained BERT encoder (frozen or fine-tuned)
  Projector : HEL → hyperbolic space
  Processor : Stack of GAA transformer layers
  Decoder   : log_0 → MLP → 3-class NLI head (Entailment / Neutral / Contradiction)

The hyperbolic representation captures entailment partial-order structure
(tree-like, delta ≈ 1.2) as discussed in Section 7.2.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from geonet.layers.hyperbolic_embedding import HyperbolicEmbeddingLayer
from geonet.layers.activations import TangentSpaceActivation, HyperbolicLayerNorm, HyperbolicDropout
from geonet.attention.geodesic_attention import GeometryAwareAttention
from geonet.utils.manifold import log_map_zero, exp_map_zero


class GeoNetNLI(nn.Module):
    """GeoNet for Natural Language Inference.

    Parameters
    ----------
    bert_model_name : str   — HuggingFace model ID (default: bert-large-uncased)
    hidden_dim      : int   — GeoNet hidden dimension (paper: 64)
    num_classes     : int   — 3 for NLI (E/N/C)
    num_layers      : int   — GAA transformer layers
    num_heads       : int   — attention heads (paper: 4)
    c_init          : float — initial curvature
    learn_c         : bool  — learnable curvature
    dropout         : float — dropout rate
    freeze_bert     : bool  — freeze BERT backbone (True = feature extraction only)
    """

    def __init__(
        self,
        bert_model_name: str = "bert-large-uncased",
        hidden_dim: int = 64,
        num_classes: int = 3,
        num_layers: int = 3,
        num_heads: int = 4,
        c_init: float = -1.0,
        learn_c: bool = True,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()

        # ── BERT encoder ──────────────────────────────────────────────────────
        self.bert_config = AutoConfig.from_pretrained(bert_model_name)
        self.bert        = AutoModel.from_pretrained(bert_model_name)
        bert_dim         = self.bert_config.hidden_size         # 1024 for large

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # ── Projection: BERT space → Poincaré ball ────────────────────────────
        self.encoder = HyperbolicEmbeddingLayer(
            bert_dim, hidden_dim, c_init=c_init, learn_c=learn_c, dropout=dropout
        )

        # ── Geometry-Aware Transformer processor ─────────────────────────────
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

        # ── NLI classification head ───────────────────────────────────────────
        # Entailment is modeled as: concat [h_prem; h_hyp; |h_prem − h_hyp|]
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @property
    def c(self) -> torch.Tensor:
        return self.encoder.c

    def _encode_sentence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of token sequences to hyperbolic sentence vectors.

        Returns
        -------
        Tensor (B, hidden_dim)  — CLS-token hyperbolic embedding
        """
        c = self.c
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state                        # (B, T, bert_dim)

        # Project full sequence to Poincaré ball
        h = self.encoder(seq_out)                                  # (B, T, hidden_dim)

        # Apply GAA layers with padding mask
        pad_mask = (attention_mask == 0)                           # True = ignore
        for gaa, norm, drop in zip(self.gaa_layers, self.norm_layers, self.drop_layers):
            # Residual connection in tangent space
            h_tan = log_map_zero(h, c)
            h_attn, _ = gaa(h, h, h, key_padding_mask=pad_mask)
            h_attn_tan = log_map_zero(h_attn, c)
            h = exp_map_zero(h_tan + h_attn_tan, c)
            h = norm(h, c)
            h = drop(h, c)

        # Pool: use CLS token position (index 0)
        return h[:, 0, :]                                          # (B, hidden_dim)

    def forward(
        self,
        prem_input_ids: torch.Tensor,
        prem_attention_mask: torch.Tensor,
        hyp_input_ids: torch.Tensor,
        hyp_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        prem_input_ids, prem_attention_mask : Tensor (B, T)
        hyp_input_ids,  hyp_attention_mask  : Tensor (B, T)

        Returns
        -------
        Tensor (B, 3)  — NLI logits [Entailment, Neutral, Contradiction]
        """
        c = self.c
        h_p = self._encode_sentence(prem_input_ids, prem_attention_mask)   # (B, d)
        h_h = self._encode_sentence(hyp_input_ids,  hyp_attention_mask)    # (B, d)

        # Project to Euclidean for final MLP
        p_e = log_map_zero(h_p, c)
        h_e = log_map_zero(h_h, c)

        # NLI features: [premise; hypothesis; |premise − hypothesis|]
        diff = (p_e - h_e).abs()
        combined = torch.cat([p_e, h_e, diff], dim=-1)            # (B, 3d)

        return self.head(combined)                                 # (B, 3)
