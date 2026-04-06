"""
train.py
─────────
Main training entry point for all GeoNet experiments.

Usage examples
--------------
  # Single run (seed 0, WordNet):
  python train.py --config configs/geonet_wordnet.yaml --seed 0

  # Full multi-seed sweep:
  for seed in {0..9}; do
      python train.py --config configs/geonet_wordnet.yaml --seed $seed
  done

  # Ablation:
  python train.py --config configs/ablations.yaml --model geonet_no_hel --seed 0

All hyperparameters are loaded from the YAML config; CLI flags override config.
Results are saved to outputs/<run_id>/results.json.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from geonet.utils.reproducibility import seed_everything, log_environment, save_results
from geonet.utils.metrics import compute_all_metrics
from geonet.optim.riemannian_adam import create_optimizers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("geonet.train")


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: Dict[str, Any] = None) -> Dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            # Support nested keys with dot notation: "model.hidden_dim"
            keys = k.split(".")
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Dict, device: torch.device) -> nn.Module:
    task     = cfg["task"]
    model_id = cfg["model"]["name"]
    mcfg     = cfg["model"]

    if task in ("graph_link", "graph_node"):
        if model_id == "geonet":
            from geonet.models.geonet_graph import GeoNetGraph
            model = GeoNetGraph(
                in_dim      = mcfg["in_dim"],
                hidden_dim  = mcfg.get("hidden_dim", 64),
                out_dim     = mcfg["out_dim"],
                num_layers  = mcfg.get("num_layers", 3),
                num_heads   = mcfg.get("num_heads", 4),
                c_init      = mcfg.get("c_init", -1.0),
                learn_c     = mcfg.get("learn_c", True),
                dropout     = mcfg.get("dropout", 0.1),
                task        = "link" if task == "graph_link" else "node",
            )
        else:
            from geonet.models.baselines import GRAPH_MODEL_REGISTRY
            cls   = GRAPH_MODEL_REGISTRY[model_id]
            model = cls(
                in_dim  = mcfg["in_dim"],
                hidden  = mcfg.get("hidden_dim", 64),
                out_dim = mcfg["out_dim"],
                dropout = mcfg.get("dropout", 0.1),
            )

    elif task == "nli":
        from geonet.models.geonet_nlp import GeoNetNLI
        if model_id == "geonet":
            model = GeoNetNLI(
                bert_model_name = mcfg.get("bert_model", "bert-large-uncased"),
                hidden_dim      = mcfg.get("hidden_dim", 64),
                num_classes     = mcfg.get("num_classes", 3),
                num_layers      = mcfg.get("num_layers", 3),
                num_heads       = mcfg.get("num_heads", 4),
                c_init          = mcfg.get("c_init", -1.0),
                learn_c         = mcfg.get("learn_c", True),
                dropout         = mcfg.get("dropout", 0.1),
                freeze_bert     = mcfg.get("freeze_bert", False),
            )
        else:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                mcfg.get("bert_model", "bert-large-uncased"),
                num_labels=mcfg.get("num_classes", 3),
            )

    elif task == "image":
        from geonet.models.geonet_vision import GeoNetVision
        if model_id == "geonet":
            model = GeoNetVision(
                num_classes    = mcfg.get("num_classes", 100),
                num_superclass = mcfg.get("num_superclass", 20),
                hidden_dim     = mcfg.get("hidden_dim", 64),
                num_layers     = mcfg.get("num_layers", 3),
                num_heads      = mcfg.get("num_heads", 4),
                c_init         = mcfg.get("c_init", -1.0),
                learn_c        = mcfg.get("learn_c", True),
                dropout        = mcfg.get("dropout", 0.1),
                backbone       = mcfg.get("backbone", "resnet101"),
                pretrained     = mcfg.get("pretrained", True),
            )
        else:
            import torchvision.models as tvm
            model = tvm.resnet101(weights=tvm.ResNet101_Weights.DEFAULT)
            model.fc = nn.Linear(2048, mcfg.get("num_classes", 100))

    else:
        raise ValueError(f"Unknown task: {task!r}")

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (dispatches to dataset-specific loaders)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg: Dict):
    """Returns (train_loader, val_loader, test_loader, dataset_meta)."""
    task    = cfg["task"]
    dataset = cfg["dataset"]["name"]
    dcfg    = cfg["dataset"]

    if dataset == "wordnet_mammals":
        from scripts.preprocess.wordnet import load_wordnet_data
        return load_wordnet_data(dcfg["path"], dcfg.get("batch_size", 256))

    elif dataset == "ogbn_arxiv":
        from scripts.preprocess.ogbn_arxiv import load_ogbn_data
        return load_ogbn_data(dcfg.get("batch_size", 256))

    elif dataset in ("snli", "multinli"):
        from scripts.preprocess.snli import load_nli_data
        return load_nli_data(dataset, dcfg.get("bert_model", "bert-large-uncased"),
                             dcfg.get("max_length", 128), dcfg.get("batch_size", 64))

    elif dataset == "cifar100":
        from scripts.preprocess.cifar100 import load_cifar100
        return load_cifar100(dcfg.get("batch_size", 256))

    elif dataset == "imagenet":
        from scripts.preprocess.imagenet import load_imagenet
        return load_imagenet(dcfg["root"], dcfg.get("batch_size", 256))

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, eucl_opt, riem_opt, criterion, device, task, cfg):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        eucl_opt.zero_grad()
        if riem_opt is not None:
            riem_opt.zero_grad()

        # Forward
        if task == "graph_node":
            logits = model(batch["x"], batch["edge_index"])
            mask   = batch.get("train_mask", torch.ones(logits.size(0), dtype=torch.bool))
            loss   = criterion(logits[mask], batch["y"][mask])

        elif task == "graph_link":
            scores = model(batch["x"], batch["edge_index"],
                           batch["src_idx"], batch["dst_idx"])
            loss   = criterion(scores, batch["labels"].float())

        elif task == "nli":
            logits = model(batch["prem_ids"], batch["prem_mask"],
                           batch["hyp_ids"],  batch["hyp_mask"])
            loss   = criterion(logits, batch["labels"])

        elif task == "image":
            out  = model(batch["images"])
            loss = criterion(out["fine_logits"], batch["labels"])
            if "super_logits" in out and "super_labels" in batch:
                loss = loss + 0.3 * criterion(out["super_logits"], batch["super_labels"])

        loss.backward()
        # Gradient clipping (prevents hyperbolic overflow)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        eucl_opt.step()
        if riem_opt is not None:
            riem_opt.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, task, dataset_meta, split="val"):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []

    for batch in tqdm(loader, desc=f"Eval [{split}]", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        if task == "graph_node":
            logits = model(batch["x"], batch["edge_index"])
            mask   = batch.get(f"{split}_mask", torch.ones(logits.size(0), dtype=torch.bool))
            all_logits.append(logits[mask].cpu())
            all_labels.append(batch["y"][mask].cpu())

        elif task == "nli" or task == "image":
            if task == "nli":
                logits = model(batch["prem_ids"], batch["prem_mask"],
                               batch["hyp_ids"],  batch["hyp_mask"])
            else:
                out = model(batch["images"])
                logits = out["fine_logits"]
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

    if not all_logits:
        return {}

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = compute_all_metrics(
        task=task,
        preds=logits,
        logits=logits,
        labels=labels,
        num_classes=dataset_meta.get("num_classes", 2),
        class_to_superclass=dataset_meta.get("class_to_superclass", {}),
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GeoNet Training")
    parser.add_argument("--config", required=True,  help="Path to YAML config")
    parser.add_argument("--seed",   type=int, default=0, help="Random seed (0-9)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--no_wandb", action="store_true")
    # Allow arbitrary config overrides: --model.hidden_dim 128
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Parse key=value overrides
    overrides = {}
    for ov in args.overrides:
        if "=" in ov:
            k, v = ov.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            overrides[k.lstrip("-")] = v

    cfg    = load_config(args.config, overrides)
    seed   = args.seed
    device = torch.device(args.device)

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed_everything(seed)
    run_id = f"{cfg['model']['name']}_{cfg['dataset']['name']}_seed{seed}"
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_environment(str(out_dir / "environment.json"))
    logger.info(f"Run: {run_id}  |  Device: {device}  |  Seed: {seed}")

    # ── WandB logging (optional) ─────────────────────────────────────────────
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project="geonet-jmlr", name=run_id, config=cfg)
        except Exception:
            logger.warning("wandb not available — skipping.")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, dataset_meta = load_data(cfg)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Optimisers ───────────────────────────────────────────────────────────
    ocfg      = cfg.get("optimizer", {})
    eucl_opt, riem_opt = create_optimizers(
        model,
        lr_eucl      = ocfg.get("lr_eucl", 1e-3),
        lr_riem      = ocfg.get("lr_riem", 3e-3),
        weight_decay = ocfg.get("weight_decay", 1e-4),
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    task = cfg["task"]
    if task == "graph_link":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # ── LR scheduler (cosine decay with warmup) ───────────────────────────────
    train_cfg  = cfg.get("training", {})
    max_epochs = train_cfg.get("epochs", 500)
    warmup_eps = train_cfg.get("warmup_epochs", 10)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        eucl_opt, T_max=max_epochs - warmup_eps, eta_min=1e-6
    )

    # ── Early stopping ────────────────────────────────────────────────────────
    patience       = train_cfg.get("patience", 50)
    best_val_score = -float("inf")
    best_epoch     = 0
    no_improve     = 0
    best_ckpt_path = out_dir / "best_model.pt"
    primary_metric = dataset_meta.get("primary_metric", "accuracy")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, eucl_opt, riem_opt,
                                 criterion, device, task, cfg)
        val_metrics = evaluate(model, val_loader, device, task, dataset_meta, split="val")

        if epoch > warmup_eps:
            scheduler.step()

        val_score = val_metrics.get(primary_metric, 0.0)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        logger.info(
            f"Epoch {epoch:4d}/{max_epochs}  "
            f"loss={train_loss:.4f}  "
            f"val_{primary_metric}={val_score:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch     = epoch
            no_improve     = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": val_metrics, "seed": seed}, best_ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info(f"Loading best checkpoint (epoch {best_epoch}) for test evaluation.")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device, task, dataset_meta, split="test")

    logger.info("=== TEST RESULTS ===")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "run_id":       run_id,
        "seed":         seed,
        "best_epoch":   best_epoch,
        "best_val":     best_val_score,
        "test_metrics": test_metrics,
        "val_metrics":  ckpt["val_metrics"],
        "n_params":     n_params,
        "config":       cfg,
        "history":      history,
    }
    save_results(results, str(out_dir / "results.json"))

    if not args.no_wandb:
        try:
            import wandb
            wandb.log({"test/" + k: v for k, v in test_metrics.items()})
            wandb.finish()
        except Exception:
            pass

    return results


if __name__ == "__main__":
    main()
