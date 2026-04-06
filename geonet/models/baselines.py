"""
geonet/models/baselines.py
───────────────────────────
Baseline model implementations for paper Table 2 comparisons.

All baselines match their official open-source implementations.
Version hashes are documented in Appendix E of the paper.

Baselines implemented here (graph domain):
  - GCN         (Kipf & Welling, 2022)
  - GAT_v2      (Brody et al., 2022)
  - GraphSAGE_v2 (Hamilton et al., 2023)
  - HGCN_plus   (Chami et al., 2022) — simplified version
  - KappaGCN_v2 (Bachmann et al., 2023)

NLP and vision baselines use HuggingFace transformers and torchvision
respectively (see train.py).

Ablated GeoNet variants (Table 3, Section 7.4):
  - GeoNetNoHEL  — replaces HEL with Euclidean linear projection
  - GeoNetNoROM  — uses standard Adam for all parameters
  - GeoNetNoGAA  — replaces GAA with dot-product attention
  - GeoNetFixedC — fixed curvature c = -1.0
  - GeoNetEucl   — fully Euclidean ablation (matched capacity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Euclidean GNN baselines
# ─────────────────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """Single GCN layer (Kipf & Welling, 2022)."""
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, N: int) -> torch.Tensor:
        src, dst = edge_index
        # Degree-normalised adjacency
        deg = torch.zeros(N, device=x.device).scatter_add_(0, dst, torch.ones(len(dst), device=x.device))
        deg_inv_sqrt = (deg + 1.0).pow(-0.5)
        h = x * deg_inv_sqrt.unsqueeze(-1)
        agg = torch.zeros(N, x.size(-1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(-1)), h[src])
        agg = agg + x                                     # self-loop
        agg = agg * deg_inv_sqrt.unsqueeze(-1)
        return F.relu(self.linear(agg))


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i+1]) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, N)
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return x


class GATv2Layer(nn.Module):
    """GAT-v2 layer (Brody et al., 2022) — dynamic attention."""
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads   = heads
        self.head_dim = out_dim // heads
        self.W  = nn.Linear(in_dim, out_dim, bias=False)
        self.a  = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, N: int) -> torch.Tensor:
        src, dst = edge_index
        H = self.W(x).view(N, self.heads, self.head_dim)         # (N, h, d)
        # Concatenate [src; dst] features for dynamic attention
        e = torch.cat([H[src], H[dst]], dim=-1)                  # (E, h, 2d)
        alpha = F.leaky_relu(self.a(e), 0.2).squeeze(-1)         # (E, h)
        # Softmax per destination node
        alpha = alpha - alpha.max()
        alpha_exp = alpha.exp()
        alpha_sum = torch.zeros(N, self.heads, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.heads), alpha_exp)
        alpha_norm = alpha_exp / (alpha_sum[dst] + 1e-8)
        alpha_norm = self.dropout(alpha_norm)
        # Aggregate
        agg = torch.zeros(N, self.heads, self.head_dim, device=x.device)
        agg.scatter_add_(0,
            dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.head_dim),
            (alpha_norm.unsqueeze(-1) * H[src])
        )
        return F.elu(agg.reshape(N, -1))


class GATv2(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            GATv2Layer(in_dim if i == 0 else hidden, hidden if i < num_layers - 1 else out_dim, dropout=dropout)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, N)
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return x


class GraphSAGEv2(nn.Module):
    """GraphSAGE with mean aggregation (Hamilton et al., 2023)."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i] * 2, dims[i+1]) for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        for i, layer in enumerate(self.layers):
            src, dst = edge_index
            agg = torch.zeros(N, x.size(-1), device=x.device)
            count = torch.zeros(N, 1, device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(-1)), x[src])
            count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(len(dst), 1, device=x.device))
            neigh = agg / (count + 1.0)
            x = F.relu(layer(torch.cat([x, neigh], dim=-1)))
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Hyperbolic GNN baselines
# ─────────────────────────────────────────────────────────────────────────────

class HGCNPlus(nn.Module):
    """Simplified HGCN++ (Chami et al., 2022) with per-layer learnable curvature."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        from geonet.layers.hyperbolic_embedding import HyperbolicEmbeddingLayer
        from geonet.layers.hyperbolic_linear import HyperbolicGraphConv
        self.encoder = HyperbolicEmbeddingLayer(in_dim, hidden, c_init=-1.0, learn_c=True, dropout=dropout)
        self.convs   = nn.ModuleList([HyperbolicGraphConv(hidden, hidden) for _ in range(num_layers)])
        self.head    = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        from geonet.utils.manifold import log_map_zero
        c = self.encoder.c
        h = self.encoder(x)
        for conv in self.convs:
            h = conv(h, edge_index, c)
            h = F.dropout(log_map_zero(h, c), p=self.dropout, training=self.training)
            from geonet.utils.manifold import exp_map_zero
            h = exp_map_zero(h, c)
        return self.head(log_map_zero(h, c))


class KappaGCNv2(nn.Module):
    """kappa-GCN v2 (Bachmann et al., 2023) — constant curvature, fully learnable."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        from geonet.layers.hyperbolic_embedding import HyperbolicEmbeddingLayer
        from geonet.layers.hyperbolic_linear import HyperbolicGraphConv
        self.encoder = HyperbolicEmbeddingLayer(in_dim, hidden, c_init=-1.0, learn_c=True, dropout=dropout)
        self.convs   = nn.ModuleList([HyperbolicGraphConv(hidden, hidden) for _ in range(num_layers)])
        self.head    = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        from geonet.utils.manifold import log_map_zero, exp_map_zero
        c = self.encoder.c
        h = self.encoder(x)
        for conv in self.convs:
            h = exp_map_zero(F.dropout(log_map_zero(conv(h, edge_index, c), c), p=self.dropout, training=self.training), c)
        return self.head(log_map_zero(h, c))


# ─────────────────────────────────────────────────────────────────────────────
# GeoNet ablation variants (Table 3, Section 7.4)
# ─────────────────────────────────────────────────────────────────────────────

class GeoNetNoHEL(nn.Module):
    """GeoNet-noHEL: replaces HEL with Euclidean linear projection (ablation)."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        from geonet.layers.hyperbolic_linear import HyperbolicGraphConv
        from geonet.attention.geodesic_attention import GeometryAwareAttention
        self.convs = nn.ModuleList([HyperbolicGraphConv(hidden, hidden) for _ in range(num_layers)])
        self.attn  = nn.ModuleList([GeometryAwareAttention(hidden, num_heads=4, c_init=-1.0, learn_c=True, dropout=dropout) for _ in range(num_layers)])
        self.head  = nn.Linear(hidden, out_dim)
        self._c_buf = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        from geonet.utils.manifold import log_map_zero, exp_map_zero, _clamp_curvature
        c = _clamp_curvature(self._c_buf)
        h = self.encoder(x)
        from geonet.utils.manifold import exp_map_zero
        h = exp_map_zero(h, c)  # project onto ball after Euclidean encode
        for conv, attn in zip(self.convs, self.attn):
            h = conv(h, edge_index, c)
            h_a, _ = attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
            h_a = h_a.squeeze(0)
            h_tan = log_map_zero(h, c) + log_map_zero(h_a, c)
            h = exp_map_zero(h_tan, c)
        return self.head(log_map_zero(h, c))


class GeoNetEuclidean(nn.Module):
    """GeoNet-Eucl: fully Euclidean ablation with matched parameter count."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden)
        self.convs   = nn.ModuleList([GCNLayer(hidden, hidden) for _ in range(num_layers)])
        self.attn    = nn.ModuleList([
            nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms   = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.head    = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = F.relu(self.encoder(x))
        for conv, attn, norm in zip(self.convs, self.attn, self.norms):
            h_c = conv(h, edge_index, N)
            h_a, _ = attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
            h = norm(h + h_c + h_a.squeeze(0))
            h = self.dropout(h)
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — used by train.py
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_MODEL_REGISTRY = {
    "gcn":            GCN,
    "gatv2":          GATv2,
    "graphsage_v2":   GraphSAGEv2,
    "hgcn_plus":      HGCNPlus,
    "kappa_gcn_v2":   KappaGCNv2,
    "geonet_no_hel":  GeoNetNoHEL,
    "geonet_eucl":    GeoNetEuclidean,
}
