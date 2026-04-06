"""
scripts/preprocess/snli.py
────────────────────────────
SNLI and MultiNLI dataset loaders — paper Section 6.1.

Downloads datasets via HuggingFace `datasets` library, tokenises with
BERT tokeniser, and returns DataLoaders matching Section 6.4 specs:
  max_length = 128, batch_size = 64.

Usage
-----
  python scripts/preprocess/snli.py --dataset snli
  python scripts/preprocess/snli.py --dataset multinli
"""

import argparse
import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Map HuggingFace label strings to indices
_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


class NLIDataset(Dataset):
    """Tokenised NLI dataset returning premise/hypothesis token pairs."""

    def __init__(self, hf_split, tokenizer, max_length: int = 128) -> None:
        self.data       = hf_split
        self.tokenizer  = tokenizer
        self.max_length = max_length
        # Filter out examples with label == -1 (annotation artefacts)
        self.valid_idx  = [
            i for i, ex in enumerate(hf_split)
            if ex["label"] in (0, 1, 2)
        ]

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ex = self.data[self.valid_idx[i]]

        prem_enc = self.tokenizer(
            ex["premise"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        hyp_enc = self.tokenizer(
            ex["hypothesis"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "prem_ids":  prem_enc["input_ids"].squeeze(0),
            "prem_mask": prem_enc["attention_mask"].squeeze(0),
            "hyp_ids":   hyp_enc["input_ids"].squeeze(0),
            "hyp_mask":  hyp_enc["attention_mask"].squeeze(0),
            "labels":    torch.tensor(ex["label"], dtype=torch.long),
        }


def load_nli_data(
    dataset_name: str = "snli",
    bert_model: str = "bert-large-uncased",
    max_length: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Load SNLI or MultiNLI and return (train, val, test) DataLoaders.

    Parameters
    ----------
    dataset_name : 'snli' or 'multinli'
    bert_model   : HuggingFace tokeniser model name
    max_length   : max token length (paper: 128)
    batch_size   : paper: 64

    Returns
    -------
    train_loader, val_loader, test_loader, meta
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install: pip install datasets transformers")

    logger.info(f"Loading {dataset_name} dataset...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    if dataset_name == "snli":
        hf_ds = load_dataset("snli")
        train_split = hf_ds["train"]
        val_split   = hf_ds["validation"]
        test_split  = hf_ds["test"]
    elif dataset_name == "multinli":
        hf_ds = load_dataset("multi_nli")
        train_split = hf_ds["train"]
        val_split   = hf_ds["validation_matched"]
        test_split  = hf_ds["validation_mismatched"]   # standard eval split
    else:
        raise ValueError(f"Unknown NLI dataset: {dataset_name!r}")

    train_ds = NLIDataset(train_split, tokenizer, max_length)
    val_ds   = NLIDataset(val_split,   tokenizer, max_length)
    test_ds  = NLIDataset(test_split,  tokenizer, max_length)

    logger.info(
        f"{dataset_name}: {len(train_ds)} train / "
        f"{len(val_ds)} val / {len(test_ds)} test examples."
    )

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)

    meta = {
        "primary_metric": "accuracy",
        "num_classes":    3,
        "label_names":    ["entailment", "neutral", "contradiction"],
    }
    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="snli", choices=["snli", "multinli"])
    args = parser.parse_args()
    load_nli_data(args.dataset)
    print(f"{args.dataset} loading verified.")
