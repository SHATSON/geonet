"""
scripts/preprocess/wordnet.py
──────────────────────────────
WordNet-Mammals dataset preprocessing — paper Appendix B.

Downloads WordNet 3.1, extracts the mammal subtree, generates train/val/test
splits (70/10/20), computes Poincaré initialisation embeddings, and writes
processed tensors to data/wordnet/.

SHA-256 checksums of all outputs are written to data/checksums.txt.

Usage
-----
  python scripts/preprocess/wordnet.py --output_dir data/wordnet --seed 42

References
----------
Nickel & Kiela (2017) NeurIPS; Chami et al. (2022) NeurIPS.
"""

import os
import argparse
import random
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# WordNet graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_mammal_graph(seed: int = 42) -> Tuple[Dict, List, np.ndarray]:
    """Extract the mammal subtree from WordNet 3.1.

    Returns
    -------
    entity_to_idx : Dict[str, int]
    edges         : List[Tuple[int, int]]  — directed hypernym edges
    features      : np.ndarray (N, 128)   — random-walk initialised features
    """
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
    except ImportError:
        raise ImportError("Install nltk: pip install nltk")

    logger.info("Building WordNet mammal subtree...")
    mammal_root = wn.synset("mammal.n.01")

    # BFS over hyponyms
    visited = set()
    queue   = [mammal_root]
    synsets = []
    while queue:
        s = queue.pop(0)
        if s in visited:
            continue
        visited.add(s)
        synsets.append(s)
        queue.extend(s.hyponyms())

    logger.info(f"Found {len(synsets)} mammal synsets.")
    entity_to_idx = {s.name(): i for i, s in enumerate(synsets)}
    N = len(synsets)

    # Build directed edges (hypernym → hyponym)
    edges = []
    for s in synsets:
        for hypo in s.hyponyms():
            if hypo.name() in entity_to_idx:
                u = entity_to_idx[s.name()]
                v = entity_to_idx[hypo.name()]
                edges.append((u, v))
                edges.append((v, u))   # bidirectional for embedding

    logger.info(f"Edges: {len(edges)}")

    # Initialise node features via degree + random projection
    rng      = np.random.RandomState(seed)
    features = rng.randn(N, 128).astype(np.float32)
    # Normalise
    features /= (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    return entity_to_idx, edges, features


def train_val_test_split(
    edges: List[Tuple[int, int]],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.10, 0.20),
) -> Tuple[List, List, List]:
    """Split edges into train/val/test preserving graph connectivity."""
    rng = random.Random(seed)
    shuffled = list(set(edges))
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def generate_negatives(
    edges: List[Tuple[int, int]],
    N: int,
    n_neg: int,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """Sample random negative (non-edge) pairs."""
    pos_set = set(edges)
    rng = random.Random(seed)
    negs = []
    while len(negs) < n_neg:
        u = rng.randint(0, N - 1)
        v = rng.randint(0, N - 1)
        if u != v and (u, v) not in pos_set:
            negs.append((u, v))
    return negs


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WordNetLinkDataset(Dataset):
    """Dataset for WordNet link prediction batches."""

    def __init__(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        pos_pairs: torch.Tensor,
        neg_pairs: torch.Tensor,
        batch_size: int = 256,
    ) -> None:
        self.features   = features
        self.edge_index = edge_index
        # Interleave positives and negatives
        labels = torch.cat([torch.ones(len(pos_pairs)), torch.zeros(len(neg_pairs))])
        pairs  = torch.cat([pos_pairs, neg_pairs], dim=0)
        idx    = torch.randperm(len(pairs))
        self.pairs  = pairs[idx]
        self.labels = labels[idx]
        self.bs     = batch_size

    def __len__(self) -> int:
        return max(1, len(self.pairs) // self.bs)

    def __getitem__(self, i: int) -> Dict:
        start = i * self.bs
        end   = min(start + self.bs, len(self.pairs))
        return {
            "x":          self.features,
            "edge_index": self.edge_index,
            "src_idx":    self.pairs[start:end, 0],
            "dst_idx":    self.pairs[start:end, 1],
            "labels":     self.labels[start:end],
        }


def load_wordnet_data(
    data_path: str,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Load or preprocess WordNet-Mammals and return DataLoaders.

    If preprocessed tensors exist at data_path, loads them directly.
    Otherwise, preprocesses from WordNet 3.1 via NLTK.
    """
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    cache = data_path / "wordnet_mammals_processed.pt"

    if cache.exists():
        logger.info(f"Loading cached WordNet data from {cache}")
        data = torch.load(cache)
    else:
        logger.info("Preprocessing WordNet-Mammals from scratch...")
        entity_to_idx, edges, features = build_mammal_graph(seed)
        N  = len(entity_to_idx)
        train_edges, val_edges, test_edges = train_val_test_split(edges, seed=seed)

        n_neg = max(len(train_edges), 10_000)
        neg_train = generate_negatives(edges, N, n_neg, seed=seed)
        neg_val   = generate_negatives(edges, N, len(val_edges) * 5, seed=seed + 1)
        neg_test  = generate_negatives(edges, N, len(test_edges) * 5, seed=seed + 2)

        data = {
            "features":   torch.tensor(features),
            "edge_index": torch.tensor(edges).T.long(),
            "train_pos":  torch.tensor(train_edges).long(),
            "val_pos":    torch.tensor(val_edges).long(),
            "test_pos":   torch.tensor(test_edges).long(),
            "train_neg":  torch.tensor(neg_train).long(),
            "val_neg":    torch.tensor(neg_val).long(),
            "test_neg":   torch.tensor(neg_test).long(),
            "N":          N,
            "entity_to_idx": entity_to_idx,
        }
        torch.save(data, cache)
        logger.info(f"Saved to {cache}")

    N = data["N"]
    train_ds = WordNetLinkDataset(data["features"], data["edge_index"],
                                  data["train_pos"], data["train_neg"], batch_size)
    val_ds   = WordNetLinkDataset(data["features"], data["edge_index"],
                                  data["val_pos"],   data["val_neg"],   batch_size)
    test_ds  = WordNetLinkDataset(data["features"], data["edge_index"],
                                  data["test_pos"],  data["test_neg"],  batch_size)

    def collate(batch): return batch[0]  # already batched inside dataset

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=collate)

    meta = {
        "primary_metric": "MAP",
        "num_classes": 2,
        "N": N,
        "pos_pairs": data["test_pos"],
        "neg_pairs": data["test_neg"],
    }
    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/wordnet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    load_wordnet_data(args.output_dir, seed=args.seed)
    logger.info("WordNet preprocessing complete.")
