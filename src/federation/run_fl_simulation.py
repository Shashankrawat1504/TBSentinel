import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import TBXRayDataset, build_eval_transform
from src.federation.client import (
    create_client_fn,
    get_parameters_from_model,
)
from src.federation.strategy import build_strategy, create_global_evaluate_fn
from src.models.image_model import TBImageClassifier

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds across libraries for reproducible FL simulation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_federated_simulation(config: DictConfig) -> dict[str, Any]:
    """Execute Flower federated learning simulation according to Hydra configuration."""
    set_seed(int(config.seed))

    device_str = str(config.device).lower()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    LOGGER.info("Running FL simulation on device: %s", device)

    # Resolve strategy configuration
    if isinstance(config.fl.strategy, (str,)):
        strategy_name = str(config.fl.strategy).lower()
        proximal_mu = 0.01 if strategy_name == "fedprox" else 0.0
    else:
        strategy_name = str(config.fl.strategy.name).lower()
        proximal_mu = (
            float(getattr(config.fl.strategy, "proximal_mu", 0.01))
            if strategy_name == "fedprox"
            else 0.0
        )

    # Resolve directories
    artifacts_dir = Path(config.paths.artifacts_dir).expanduser().resolve()
    checkpoints_dir = Path(config.paths.checkpoints_dir).expanduser().resolve()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(config.data.federated_splits_dir).expanduser().resolve()
    val_csv = Path(config.data.val_csv).expanduser().resolve()
    image_root = Path(config.data.image_root).expanduser().resolve()

    num_clients = int(config.fl.num_clients)
    for cid in range(num_clients):
        client_csv = splits_dir / f"client_{cid}_train.csv"
        if not client_csv.is_file():
            raise FileNotFoundError(f"Missing partition CSV for client {cid}: {client_csv}")

    if not val_csv.is_file():
        raise FileNotFoundError(f"Missing global validation CSV: {val_csv}")

    # Build fixed global validation DataLoader (1800 samples)
    eval_transform = build_eval_transform(image_size=int(config.model.image_size))
    val_dataset = TBXRayDataset(
        annotations=val_csv,
        image_root=image_root,
        transform=eval_transform,
        image_size=int(config.model.image_size),
    )
    val_batch_size = int(config.fl.evaluation.get("batch_size", 16))
    val_workers = int(config.fl.evaluation.get("num_workers", 0))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    LOGGER.info("Loaded global validation set: %d samples", len(val_dataset))

    # Optional WandB logging
    wandb_tracker = None
    if getattr(config.fl.wandb, "enabled", False):
        wandb_config = OmegaConf.to_container(config, resolve=True)
        if not isinstance(wandb_config, dict):
            raise TypeError("Resolved Hydra configuration must be a mapping.")
        wandb_mode = str(config.fl.wandb.mode)
        supported_wandb_modes = {"online", "offline", "disabled", "shared"}
        if wandb_mode not in supported_wandb_modes:
            raise ValueError(
                f"Unsupported W&B mode {wandb_mode!r}; "
                f"expected one of {sorted(supported_wandb_modes)}."
            )
        validated_wandb_mode = cast(
            Literal["online", "offline", "disabled", "shared"],
            wandb_mode,
        )

        try:
            import wandb

            wandb_run = wandb.init(
                project=str(config.project_name),
                name=f"fl-{strategy_name}-r{config.fl.num_rounds}-e{config.fl.local_epochs}",
                config={str(key): value for key, value in wandb_config.items()},
                mode=validated_wandb_mode,
                reinit=True,
            )
            wandb_tracker = wandb_run
            LOGGER.info("WandB initialized (mode=%s)", config.fl.wandb.mode)
        except Exception as e:
            LOGGER.warning("WandB initialization skipped: %s", e)

    # Initialize global model (keep on CPU initially to keep GPU free for clients)
    global_model = TBImageClassifier(
        num_classes=int(config.model.num_classes),
        pretrained=bool(config.model.pretrained),
        freeze_backbone=bool(config.model.freeze_backbone),
        dropout=float(config.model.dropout),
    )

    initial_ndarrays = get_parameters_from_model(global_model)
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    # Checkpoint filenames
    best_ckpt_name = f"tb_federated_{strategy_name}_best.pt"
    last_ckpt_name = f"tb_federated_{strategy_name}_last.pt"

    history_tracker: list[dict[str, Any]] = []
    evaluate_fn = create_global_evaluate_fn(
        model=global_model,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoints_dir,
        best_checkpoint_name=best_ckpt_name,
        last_checkpoint_name=last_ckpt_name,
        class_names=tuple(config.data.training.label_names),
        history_tracker=history_tracker,
        wandb_tracker=wandb_tracker,
    )

    # Build Federated strategy
    strategy = build_strategy(
        strategy_name=strategy_name,
        fraction_fit=float(config.fl.fraction_fit),
        fraction_evaluate=float(config.fl.fraction_evaluate),
        min_fit_clients=int(config.fl.min_fit_clients),
        min_evaluate_clients=int(config.fl.min_evaluate_clients),
        min_available_clients=int(config.fl.min_available_clients),
        proximal_mu=proximal_mu,
        local_epochs=int(config.fl.local_epochs),
        learning_rate=float(config.fl.learning_rate),
        weight_decay=float(config.fl.weight_decay),
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
    )

    # Build client factory
    client_fn = create_client_fn(
        federated_splits_dir=splits_dir,
        image_root=image_root,
        image_size=int(config.model.image_size),
        batch_size=int(config.fl.batch_size),
        num_workers=0,  # 0 is safe for simulation actors on Windows
        device=device,
        learning_rate=float(config.fl.learning_rate),
        weight_decay=float(config.fl.weight_decay),
        local_epochs=int(config.fl.local_epochs),
        proximal_mu=proximal_mu,
        mixed_precision=bool(config.fl.mixed_precision),
        model_kwargs={
            "num_classes": int(config.model.num_classes),
            "pretrained": False,
            "freeze_backbone": bool(config.model.freeze_backbone),
            "dropout": float(config.model.dropout),
        },
    )

    client_resources = OmegaConf.to_container(config.fl.client_resources, resolve=True)
    num_rounds = int(config.fl.num_rounds)

    LOGGER.info(
        "Starting Flower simulation: strategy=%s, rounds=%d, local_epochs=%d, clients=%d, proximal_mu=%.4f",
        strategy_name,
        num_rounds,
        int(config.fl.local_epochs),
        num_clients,
        proximal_mu,
    )

    # Run Flower simulation
    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    best_checkpoint_path = checkpoints_dir / best_ckpt_name
    last_checkpoint_path = checkpoints_dir / last_ckpt_name

    # Load and evaluate best global model
    best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    global_model.load_state_dict(best_ckpt["model_state_dict"])
    final_metrics = best_ckpt["metrics"]

    LOGGER.info(
        "=== Final Best Federated Model (%s) ===",
        strategy_name.upper(),
    )
    LOGGER.info("Best Round: %s", best_ckpt.get("round", "N/A"))
    LOGGER.info("Accuracy:    %.4f", final_metrics["accuracy"])
    LOGGER.info("Macro-F1:    %.4f", final_metrics["macro_f1"])
    LOGGER.info("AUC-ROC:     %.4f", final_metrics["auc"])
    LOGGER.info("Sensitivity: %.4f", final_metrics["sensitivity"])
    LOGGER.info("Specificity: %.4f", final_metrics["specificity"])
    LOGGER.info("Confusion:   %s", final_metrics["confusion_matrix"])

    # Save summary report JSON
    summary_data = {
        "strategy": strategy_name,
        "proximal_mu": proximal_mu if strategy_name == "fedprox" else 0.0,
        "num_rounds": num_rounds,
        "local_epochs": int(config.fl.local_epochs),
        "batch_size": int(config.fl.batch_size),
        "learning_rate": float(config.fl.learning_rate),
        "best_round": best_ckpt.get("round"),
        "best_metrics": final_metrics,
        "history": history_tracker,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
    }

    output_dir = artifacts_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"federated_{strategy_name}_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    LOGGER.info("Saved federated summary results to: %s", summary_file)

    if wandb_tracker is not None:
        try:
            wandb_tracker.finish()
        except Exception:
            pass

    return summary_data


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main CLI entrypoint for Hydra-configured federated simulation."""
    run_federated_simulation(config)


if __name__ == "__main__":
    main()
