"""
geonet/utils/metrics.py
───────────────────────
All evaluation metrics used in the paper (Section 6.3).

Metrics
-------
- mean_average_precision  — MAP for link prediction (WordNet, Table 2)
- fermi_dirac_accuracy    — Fermi-Dirac link prediction accuracy
- embedding_distortion    — geometric quality metric (Table 2)
- hierarchical_precision  — HP@k for hierarchical label evaluation (Table 2)
- accuracy                — flat classification accuracy
- macro_f1                — macro-averaged F1 (NLI tasks)
- compute_all_metrics     — dispatcher used by evaluate.py

References
----------
Nickel & Kiela (2017); Wu et al. (2023); Sarkar (2011).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import f1_score, average_precision_score

from geonet.utils.manifold import geodesic_distance, _EPS


# ─────────────────────────────────────────────────────────────────────────────
# Graph / link-prediction metrics
# ─────────────────────────────────────────────────────────────────────────────

def mean_average_precision(
    embeddings: torch.Tensor,
    pos_pairs: torch.Tensor,
    neg_pairs: torch.Tensor,
    c: torch.Tensor,
) -> float:
    """Mean Average Precision for link prediction.

    For each positive pair (i, j), rank all negative pairs by distance and
    compute AP.  Aggregated over all positives to give MAP (paper Table 2).

    Parameters
    ----------
    embeddings : Tensor (N, d)  — hyperbolic embeddings
    pos_pairs  : Tensor (P, 2)  — indices of positive (linked) pairs
    neg_pairs  : Tensor (Q, 2)  — indices of negative (unlinked) pairs
    c          : Tensor ()      — curvature

    Returns
    -------
    float — MAP score in [0, 1].
    """
    embeddings = embeddings.detach()
    aps = []
    for i, j in pos_pairs:
        d_pos = geodesic_distance(
            embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0), c
        ).item()
        d_neg = geodesic_distance(
            embeddings[neg_pairs[:, 0]], embeddings[neg_pairs[:, 1]], c
        ).detach().cpu().numpy()
        # Lower distance = more likely linked → scores are negated distances
        scores = np.concatenate([[-d_pos], -d_neg])
        labels = np.concatenate([[1], np.zeros(len(d_neg))])
        aps.append(average_precision_score(labels, scores))
    return float(np.mean(aps))


def fermi_dirac_accuracy(
    embeddings: torch.Tensor,
    pos_pairs: torch.Tensor,
    neg_pairs: torch.Tensor,
    c: torch.Tensor,
    r: float = 2.0,
    t: float = 1.0,
) -> float:
    """Fermi-Dirac link prediction accuracy (Nickel & Kiela, 2017).

    P(edge | i,j) = 1 / (exp((d(i,j) − r) / t) + 1)
    Threshold at 0.5 and compute accuracy.
    """
    embeddings = embeddings.detach()
    all_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
    labels = torch.cat([
        torch.ones(len(pos_pairs)),
        torch.zeros(len(neg_pairs))
    ])
    d = geodesic_distance(
        embeddings[all_pairs[:, 0]], embeddings[all_pairs[:, 1]], c
    ).detach().cpu()
    probs = 1.0 / (torch.exp((d - r) / t) + 1.0)
    preds = (probs > 0.5).float()
    return (preds == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Geometric quality
# ─────────────────────────────────────────────────────────────────────────────

def compute_distortion(
    embeddings: torch.Tensor,
    true_dist_matrix: torch.Tensor,
    c: torch.Tensor,
    n_samples: int = 10_000,
    seed: int = 42,
) -> float:
    """Mean embedding distortion over randomly sampled point pairs.

    distortion = E_pairs[ |d_hyperbolic(i,j) / d_true(i,j) − 1| ]

    Reports the values in Table 2 (Section 7.1).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    N = embeddings.size(0)
    idx = torch.randint(0, N, (n_samples * 2,), generator=rng).reshape(-1, 2)
    mask = idx[:, 0] != idx[:, 1]
    idx = idx[mask][:n_samples]
    i, j = idx[:, 0], idx[:, 1]

    d_emb  = geodesic_distance(embeddings[i].detach(), embeddings[j].detach(), c)
    d_true = true_dist_matrix[i, j]
    ratio  = d_emb / (d_true + _EPS)
    return (ratio - 1.0).abs().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 flat classification accuracy."""
    if preds.dim() > 1:
        preds = preds.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy."""
    top_k = logits.topk(k, dim=-1).indices
    correct = top_k.eq(labels.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1 score (used for NLI evaluation, Section 6.3)."""
    if preds.dim() > 1:
        preds = preds.argmax(dim=-1)
    return f1_score(
        labels.cpu().numpy(),
        preds.cpu().numpy(),
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0,
    )


def hierarchical_precision_at_k(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_to_superclass: Dict[int, int],
    k: int = 5,
) -> float:
    """Hierarchical Precision HP@k (Wu et al., 2023).

    Fraction of top-k predictions that share a superclass with the ground-truth
    label.  Used for CIFAR-100 and ImageNet results in Section 7.3.

    Parameters
    ----------
    logits            : Tensor (B, C)
    labels            : Tensor (B,)
    class_to_superclass: Dict mapping fine class index → superclass index
    k                 : int

    Returns
    -------
    float — HP@k in [0, 1].
    """
    top_k = logits.topk(k, dim=-1).indices.cpu().numpy()   # (B, k)
    labels_np = labels.cpu().numpy()                        # (B,)
    scores = []
    for b in range(len(labels_np)):
        true_super = class_to_superclass.get(int(labels_np[b]), -1)
        pred_supers = [class_to_superclass.get(int(p), -2) for p in top_k[b]]
        scores.append(float(true_super in pred_supers))
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    task: str,
    **kwargs,
) -> Dict[str, float]:
    """Compute all metrics for a given task.

    Parameters
    ----------
    task : str  — one of 'graph_link', 'graph_node', 'nli', 'image'
    **kwargs    — task-specific arguments

    Returns
    -------
    Dict mapping metric name → value.
    """
    if task == "graph_link":
        return {
            "MAP":      mean_average_precision(**{k: kwargs[k] for k in ("embeddings", "pos_pairs", "neg_pairs", "c")}),
            "FD_acc":   fermi_dirac_accuracy(**{k: kwargs[k] for k in ("embeddings", "pos_pairs", "neg_pairs", "c")}),
            "distortion": compute_distortion(**{k: kwargs[k] for k in ("embeddings", "true_dist_matrix", "c")}),
        }
    elif task == "graph_node":
        return {
            "accuracy":  accuracy(kwargs["preds"], kwargs["labels"]),
            "distortion": compute_distortion(**{k: kwargs[k] for k in ("embeddings", "true_dist_matrix", "c")}),
        }
    elif task == "nli":
        return {
            "accuracy": accuracy(kwargs["preds"], kwargs["labels"]),
            "macro_f1": macro_f1(kwargs["preds"], kwargs["labels"], kwargs["num_classes"]),
        }
    elif task == "image":
        result = {
            "top1_accuracy": accuracy(kwargs["logits"], kwargs["labels"]),
            "top5_accuracy": top_k_accuracy(kwargs["logits"], kwargs["labels"], k=5),
        }
        if "class_to_superclass" in kwargs:
            result["HP@5"] = hierarchical_precision_at_k(
                kwargs["logits"], kwargs["labels"],
                kwargs["class_to_superclass"], k=5,
            )
        return result
    else:
        raise ValueError(f"Unknown task: {task!r}")
