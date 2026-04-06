"""
scripts/preprocess/ogbn_arxiv.py
──────────────────────────────────
ogbn-arxiv dataset loader — paper Section 6.1.

Uses the official OGB library (v1.3.6) to download and load the ogbn-arxiv
citation graph.  Node features are 128-dim paper2vec embeddings.
Official OGB train/val/test splits are used without modification.

Usage
-----
  python scripts/preprocess/ogbn_arxiv.py
"""

import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class OGBNNodeDataset(Dataset):
    """Wraps OGB graph data as a single-batch node classification dataset."""

    def __init__(self, data, split_idx: Dict, split: str) -> None:
        self.data      = data
        self.mask      = split_idx[split]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _) -> Dict:
        return {
            "x":          self.data.x.float(),
            "edge_index": self.data.edge_index,
            "y":          self.data.y.squeeze(-1),
            f"{self._split}_mask": self.mask,
        }

    @property
    def _split(self) -> str:
        # Recover split name from mask contents (used for metric masking)
        return "train"  # overridden per-split below


class _SplitDataset(Dataset):
    def __init__(self, x, edge_index, y, mask, split_name):
        self.x = x; self.edge_index = edge_index
        self.y = y; self.mask = mask; self.split_name = split_name

    def __len__(self): return 1

    def __getitem__(self, _):
        return {
            "x":          self.x,
            "edge_index": self.edge_index,
            "y":          self.y,
            f"{self.split_name}_mask": self.mask,
        }


def load_ogbn_data(
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Download (if needed) and load ogbn-arxiv via OGB.

    Returns standard (train, val, test) DataLoader tuple with dataset metadata.
    """
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        import torch_geometric
    except ImportError:
        raise ImportError(
            "Install OGB and PyG: pip install ogb torch-geometric"
        )

    logger.info("Loading ogbn-arxiv (OGB v1.3.6)...")
    dataset   = PygNodePropPredDataset(name="ogbn-arxiv", root="data/ogbn")
    data      = dataset[0]
    split_idx = dataset.get_idx_split()

    x          = data.x.float()                    # (N, 128)
    edge_index = data.edge_index                   # (2, E)
    y          = data.y.squeeze(-1)                # (N,)

    def collate(batch): return batch[0]

    def make_loader(split):
        ds = _SplitDataset(x, edge_index, y, split_idx[split], split)
        return DataLoader(ds, batch_size=1, shuffle=(split == "train"), collate_fn=collate)

    train_loader = make_loader("train")
    val_loader   = make_loader("valid")
    test_loader  = make_loader("test")

    meta = {
        "primary_metric": "accuracy",
        "num_classes":    dataset.num_classes,   # 40
        "class_to_superclass": {},               # no superclass for ogbn
    }
    logger.info(f"ogbn-arxiv: {x.size(0)} nodes, {edge_index.size(1)} edges, "
                f"{dataset.num_classes} classes.")
    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    load_ogbn_data()
    print("ogbn-arxiv loading verified.")
