"""
scripts/preprocess/cifar100.py
────────────────────────────────
CIFAR-100 dataset loader with superclass hierarchy — paper Section 6.1.

Provides:
  - Standard torchvision CIFAR-100 DataLoaders
  - class_to_superclass mapping (fine-class → coarse-class index)
    used for HP@5 hierarchical precision evaluation (Section 7.3)
  - Data augmentation matching Section 6.4 (paper normalisation values)

Usage
-----
  python scripts/preprocess/cifar100.py
"""

import logging
from typing import Dict, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-100 fine-to-coarse (superclass) mapping
# 20 superclasses × 5 fine classes each = 100 classes
# Indices follow torchvision CIFAR-100 class ordering.
# ─────────────────────────────────────────────────────────────────────────────

# fmt: off
CIFAR100_SUPERCLASS_MAP: Dict[int, int] = {
    # aquatic mammals (0)
    4:0, 30:0, 55:0, 72:0, 95:0,
    # fish (1)
    1:1, 32:1, 67:1, 73:1, 91:1,
    # flowers (2)
    54:2, 62:2, 70:2, 82:2, 92:2,
    # food containers (3)
    9:3, 10:3, 16:3, 28:3, 61:3,
    # fruit and vegetables (4)
    0:4, 51:4, 53:4, 57:4, 83:4,
    # household electrical devices (5)
    22:5, 39:5, 40:5, 86:5, 87:5,
    # household furniture (6)
    5:6, 20:6, 25:6, 84:6, 94:6,
    # insects (7)
    6:7, 7:7, 14:7, 18:7, 24:7,
    # large carnivores (8)
    3:8, 42:8, 43:8, 88:8, 97:8,
    # large man-made outdoor things (9)
    12:9, 17:9, 37:9, 68:9, 76:9,
    # large natural outdoor scenes (10)
    23:10, 33:10, 49:10, 60:10, 71:10,
    # large omnivores and herbivores (11)
    15:11, 19:11, 21:11, 31:11, 38:11,
    # medium-sized mammals (12)
    34:12, 63:12, 64:12, 66:12, 75:12,
    # non-insect invertebrates (13)
    26:13, 45:13, 77:13, 79:13, 99:13,
    # people (14)
    2:14, 11:14, 35:14, 46:14, 98:14,
    # reptiles (15)
    27:15, 29:15, 44:15, 78:15, 93:15,
    # small mammals (16)
    36:16, 50:16, 65:16, 74:16, 80:16,
    # trees (17)
    47:17, 52:17, 56:17, 59:17, 96:17,
    # vehicles 1 (18)
    8:18, 13:18, 48:18, 58:18, 90:18,
    # vehicles 2 (19)
    41:19, 69:19, 81:19, 85:19, 89:19,
}
# fmt: on


def load_cifar100(
    batch_size: int = 256,
    num_workers: int = 4,
    data_root: str = "data/cifar100",
    # Paper Section 6.4 normalisation values
    mean: Tuple = (0.5071, 0.4867, 0.4408),
    std:  Tuple = (0.2675, 0.2565, 0.2761),
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Load CIFAR-100 with standard augmentation.

    Train augmentation: RandomCrop(32, pad=4) + RandomHorizontalFlip + Normalise
    Val/Test:           Normalise only

    Returns
    -------
    train_loader, val_loader, test_loader, meta
    """
    normalise = T.Normalize(mean=mean, std=std)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalise,
    ])
    test_transform = T.Compose([T.ToTensor(), normalise])

    train_ds = torchvision.datasets.CIFAR100(
        root=data_root, train=True,  download=True, transform=train_transform
    )
    test_ds  = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=test_transform
    )

    # Carve 5 000 examples from training set for validation (fixed split)
    # Deterministic split: first 45 000 train, last 5 000 val
    val_ds   = torch.utils.data.Subset(train_ds, range(45_000, 50_000))
    train_ds = torch.utils.data.Subset(train_ds, range(45_000))

    logger.info(
        f"CIFAR-100: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test."
    )

    pin_mem = torch.cuda.is_available()

    def _make_batch(batch):
        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch])
        super_labels = torch.tensor([
            CIFAR100_SUPERCLASS_MAP[int(l)] for l in labels
        ])
        return {"images": images, "labels": labels, "super_labels": super_labels}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem,
                              collate_fn=_make_batch)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem,
                              collate_fn=_make_batch)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem,
                              collate_fn=_make_batch)

    meta = {
        "primary_metric":      "top1_accuracy",
        "num_classes":         100,
        "num_superclass":      20,
        "class_to_superclass": CIFAR100_SUPERCLASS_MAP,
    }
    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_cifar100()
    print("CIFAR-100 loading verified.")
