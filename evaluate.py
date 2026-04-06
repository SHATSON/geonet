"""
evaluate.py
────────────
Standalone evaluation script.

Loads a saved checkpoint and computes all test-set metrics referenced
in the paper, including embedding distortion (requires graph data).

Usage
-----
  python evaluate.py \
      --checkpoint outputs/geonet_wordnet_mammals_seed0/best_model.pt \
      --config     configs/geonet_wordnet.yaml \
      --split      test

To aggregate over all 10 seeds and produce mean ± std (as in Table 2):
  python evaluate.py --config configs/geonet_wordnet.yaml --all_seeds
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np

from geonet.utils.reproducibility import seed_everything, load_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("geonet.evaluate")


def evaluate_checkpoint(checkpoint_path: str, config_path: str, split: str = "test", device: str = "cpu"):
    from train import load_config, build_model, load_data
    from geonet.utils.metrics import compute_all_metrics

    cfg = load_config(config_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    seed = ckpt.get("seed", 0)
    seed_everything(seed)

    device = torch.device(device)
    model  = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_loader, test_loader, meta = load_data(cfg)
    loader = test_loader if split == "test" else val_loader

    from train import evaluate as _evaluate
    metrics = _evaluate(model, loader, device, cfg["task"], meta, split=split)
    logger.info(f"[Seed {seed}] {split} metrics: {metrics}")
    return metrics


def aggregate_seeds(config_path: str, output_root: str = "outputs", split: str = "test"):
    """Aggregate results from seeds 0-9 and report mean ± std."""
    from train import load_config
    cfg = load_config(config_path)
    model_name   = cfg["model"]["name"]
    dataset_name = cfg["dataset"]["name"]

    all_metrics = {}
    for seed in range(10):
        run_id = f"{model_name}_{dataset_name}_seed{seed}"
        results_path = Path(output_root) / run_id / "results.json"
        if not results_path.exists():
            logger.warning(f"Missing results for seed {seed}: {results_path}")
            continue
        r = load_results(str(results_path))
        for k, v in r["test_metrics"].items():
            all_metrics.setdefault(k, []).append(v)

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name} on {dataset_name}  (n={len(next(iter(all_metrics.values())))} seeds)")
    print(f"{'='*60}")
    summary = {}
    for k, vals in all_metrics.items():
        arr  = np.array(vals)
        mean = arr.mean()
        std  = arr.std()
        ci95 = 1.96 * std / np.sqrt(len(arr))
        print(f"  {k:30s}: {mean:.4f} ± {std:.4f}  (95% CI ±{ci95:.4f})")
        summary[k] = {"mean": float(mean), "std": float(std), "ci95": float(ci95), "n": len(vals)}
    print(f"{'='*60}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="GeoNet Evaluation")
    parser.add_argument("--checkpoint", help="Path to checkpoint file")
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--all_seeds",  action="store_true",
                        help="Aggregate over all 10 seeds from outputs/")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.all_seeds:
        aggregate_seeds(args.config, args.output_root, args.split)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint required unless --all_seeds is set.")
        metrics = evaluate_checkpoint(args.checkpoint, args.config, args.split, args.device)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
