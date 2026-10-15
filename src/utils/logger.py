
# src/utils/logger.py

import logging
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


LOGGER_NAME = "TBSentinel"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = LOGGER_NAME,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or retrieve a logger with console and optional file output.

    Repeated calls do not add duplicate handlers. Handlers owned by
    this utility can be safely reused or updated.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # Reuse this utility's console handler if already configured.
    console_handler = next(
        (
            handler
            for handler in logger.handlers
            if getattr(handler, "_tbsentinel_console", False)
        ),
        None,
    )

    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler._tbsentinel_console = True
        logger.addHandler(console_handler)

    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Configure an optional file handler.
    if log_file is not None:
        file_path = Path(log_file).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = next(
            (
                handler
                for handler in logger.handlers
                if getattr(handler, "_tbsentinel_file", None)
                == str(file_path)
            ),
            None,
        )

        if file_handler is None:
            file_handler = logging.FileHandler(
                file_path,
                encoding="utf-8",
            )
            file_handler._tbsentinel_file = str(file_path)
            logger.addHandler(file_handler)

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

    return logger


def init_wandb(
    cfg: DictConfig,
    job_type: str = "train",
    enabled: bool = True,
    mode: str | None = None,
) -> Any | None:
    """
    Initialize optional Weights & Biases experiment tracking.

    Args:
        cfg: Resolved Hydra configuration.
        job_type: Experiment type, e.g. train, evaluate, or federated.
        enabled: Disable tracking when False.
        mode: Optional W&B mode, e.g. offline or disabled.

    Returns:
        The W&B run, or None if tracking is disabled.
    """
    if not enabled or mode == "disabled":
        return None

    logger = get_logger()

    try:
        from omegaconf import OmegaConf
        import wandb

        config_dict = OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
        )

        if not isinstance(config_dict, dict):
            raise ValueError("Expected the root Hydra configuration to be a mapping.")

        project_name = cfg.get("project_name", LOGGER_NAME)
        experiment_name = cfg.get("experiment_name", job_type)

        kwargs = {
            "project": project_name,
            "name": experiment_name,
            "config": config_dict,
            "job_type": job_type,
            "reinit": True,
        }

        if mode is not None:
            kwargs["mode"] = mode

        run = wandb.init(**kwargs)
        logger.info(
            "W&B initialized: project=%s, job_type=%s",
            project_name,
            job_type,
        )
        return run

    except Exception:
        logger.exception("Failed to initialize W&B tracking.")
        raise


def log_metrics(
    metrics: dict[str, Any],
    step: int | None = None,
) -> None:
    """Log scalar metrics to the active W&B run, if available."""
    if not metrics:
        return

    import wandb

    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb() -> None:
    """Finish the active W&B run if one exists."""
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        wandb.finish()


def close_logger(logger: logging.Logger) -> None:
    """
    Close and remove handlers owned by this utility.

    Call this when the logger is no longer needed, not after every
    training epoch.
    """
    for handler in logger.handlers[:]:
        if (
            getattr(handler, "_tbsentinel_console", False)
            or getattr(handler, "_tbsentinel_file", None) is not None
        ):
            logger.removeHandler(handler)
            handler.close()